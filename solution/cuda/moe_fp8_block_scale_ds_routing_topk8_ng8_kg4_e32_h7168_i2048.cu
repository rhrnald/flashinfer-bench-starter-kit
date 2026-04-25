#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "reference_backend.h"
#include "moe_tc_backend_b200_step1_direct.cuh"
#include "moe_tc_backend_b200_step2_direct.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <vector>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

using tvm::ffi::TensorView;

namespace {

bool UseDirectImplementation() {
  const char* impl = std::getenv("FIB_MOE_IMPL");
  if (impl == nullptr) return false;
  return std::strcmp(impl, "direct") == 0;
}

constexpr int kNumExperts = 256;
constexpr int kNumLocalExperts = 32;
constexpr int kHidden = 7168;
constexpr int kIntermediate = 2048;
constexpr int kBlock = 128;
constexpr int kNumGroups = 8;
constexpr int kGroupSize = 32;
constexpr int kTopKGroups = 4;
constexpr int kTopK = 8;

void* g_step1_comm_w13_tma_desc_dev = nullptr;
void* g_step1_direct_w13_tma_desc_dev = nullptr;
void* g_step2_w2_tma_desc_dev = nullptr;
void* g_step2_w2_tma_desc_k4_dev = nullptr;
void* g_cached_direct_gemm1_ptr = nullptr;
void* g_cached_direct_gemm2_ptr = nullptr;
bool g_cached_direct_tma = false;
bool g_cached_direct_tma_sw128 = false;

int ParseStep2PipelineVariant() {
  const char* env = std::getenv("FIB_STEP2_PIPELINE_VARIANT");
  if (env == nullptr || std::strcmp(env, "m64n8_k8_db_batch2") == 0) {
    return direct_backend::kStep2PipeM64N8K8DbBatch2;
  }
  if (std::strcmp(env, "m64n8_k4_db_batch2") == 0) {
    return direct_backend::kStep2PipeM64N8K4DbBatch2;
  }
  if (std::strcmp(env, "m64n16_k8_db_batch2") == 0) {
    // N16 is kept as a selector alias for sweep bookkeeping. A real N16 path
    // needs wider partial/scaled TMEM than the current safe Step2 allocation.
    return direct_backend::kStep2PipeM64N16K8DbBatch2;
  }
  if (std::strcmp(env, "m64n16_k4_db_batch2") == 0) {
    // See N16 note above; this currently exercises the K4 pipeline shape.
    return direct_backend::kStep2PipeM64N16K4DbBatch2;
  }
  if (std::strcmp(env, "m128n8_k8_db_batch2") == 0) {
    // M128 is kept as a selector alias. A real path would exceed the current
    // double-buffered W2 shared-memory budget without a separate design.
    return direct_backend::kStep2PipeM128N8K8DbBatch2;
  }
  return direct_backend::kStep2PipeM64N8K8DbBatch2;
}

inline void CheckCuLocal(CUresult status, const char* where) {
  if (status == CUDA_SUCCESS) return;
  const char* msg = nullptr;
  cuGetErrorString(status, &msg);
  if (msg == nullptr) msg = "unknown CUresult";
  TVM_FFI_ICHECK_EQ(status, CUDA_SUCCESS) << where << ": " << msg;
}

inline void CheckCudaLocal(cudaError_t status, const char* where) {
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << where << ": " << cudaGetErrorString(status);
}

inline CUtensorMap EncodeTensorMap2DUint8Local(uint8_t* base, uint64_t dim0, uint64_t dim1,
                                               uint64_t stride1_elems, uint32_t box0,
                                               uint32_t box1,
                                               CUtensorMapSwizzle smem_swizzle =
                                                   CU_TENSOR_MAP_SWIZZLE_NONE) {
  CUtensorMap out{};
  uint64_t global_dims[2] = {dim0, dim1};
  uint64_t global_strides[1] = {stride1_elems * sizeof(uint8_t)};
  uint32_t box_dims[2] = {box0, box1};
  uint32_t elem_strides[2] = {1, 1};
  CheckCuLocal(cuTensorMapEncodeTiled(
                   &out, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, reinterpret_cast<void*>(base),
                   global_dims, global_strides, box_dims, elem_strides,
                   CU_TENSOR_MAP_INTERLEAVE_NONE, smem_swizzle,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(step1 comm w13)");
  return out;
}

inline CUtensorMap EncodeTensorMap3DUint8Local(uint8_t* base, uint64_t dim0, uint64_t dim1,
                                               uint64_t dim2, uint64_t stride1_elems,
                                               uint64_t stride2_elems, uint32_t box0,
                                               uint32_t box1, uint32_t box2,
                                               CUtensorMapSwizzle smem_swizzle =
                                                   CU_TENSOR_MAP_SWIZZLE_NONE) {
  CUtensorMap out{};
  uint64_t global_dims[3] = {dim0, dim1, dim2};
  uint64_t global_strides[2] = {stride1_elems * sizeof(uint8_t),
                                stride2_elems * sizeof(uint8_t)};
  uint32_t box_dims[3] = {box0, box1, box2};
  uint32_t elem_strides[3] = {1, 1, 1};
  CheckCuLocal(cuTensorMapEncodeTiled(
                   &out, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, reinterpret_cast<void*>(base),
                   global_dims, global_strides, box_dims, elem_strides,
                   CU_TENSOR_MAP_INTERLEAVE_NONE, smem_swizzle,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(step1 comm w13 3d)");
  return out;
}

inline CUtensorMap EncodeTensorMap4DUint8Local(uint8_t* base, uint64_t dim0, uint64_t dim1,
                                               uint64_t dim2, uint64_t dim3,
                                               uint64_t stride1_elems,
                                               uint64_t stride2_elems,
                                               uint64_t stride3_elems, uint32_t box0,
                                               uint32_t box1, uint32_t box2,
                                               uint32_t box3,
                                               CUtensorMapSwizzle smem_swizzle =
                                                   CU_TENSOR_MAP_SWIZZLE_NONE) {
  CUtensorMap out{};
  uint64_t global_dims[4] = {dim0, dim1, dim2, dim3};
  uint64_t global_strides[3] = {stride1_elems * sizeof(uint8_t),
                                stride2_elems * sizeof(uint8_t),
                                stride3_elems * sizeof(uint8_t)};
  uint32_t box_dims[4] = {box0, box1, box2, box3};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CheckCuLocal(cuTensorMapEncodeTiled(
                   &out, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, reinterpret_cast<void*>(base),
                   global_dims, global_strides, box_dims, elem_strides,
                   CU_TENSOR_MAP_INTERLEAVE_NONE, smem_swizzle,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(step1 comm w13 4d)");
  return out;
}

inline CUtensorMap EncodeTensorMap5DUint8Local(uint8_t* base, uint64_t dim0, uint64_t dim1,
                                               uint64_t dim2, uint64_t dim3, uint64_t dim4,
                                               uint64_t stride1_elems,
                                               uint64_t stride2_elems,
                                               uint64_t stride3_elems,
                                               uint64_t stride4_elems, uint32_t box0,
                                               uint32_t box1, uint32_t box2,
                                               uint32_t box3, uint32_t box4,
                                               CUtensorMapSwizzle smem_swizzle =
                                                   CU_TENSOR_MAP_SWIZZLE_NONE) {
  CUtensorMap out{};
  uint64_t global_dims[5] = {dim0, dim1, dim2, dim3, dim4};
  uint64_t global_strides[4] = {stride1_elems * sizeof(uint8_t),
                                stride2_elems * sizeof(uint8_t),
                                stride3_elems * sizeof(uint8_t),
                                stride4_elems * sizeof(uint8_t)};
  uint32_t box_dims[5] = {box0, box1, box2, box3, box4};
  uint32_t elem_strides[5] = {1, 1, 1, 1, 1};
  CheckCuLocal(cuTensorMapEncodeTiled(
                   &out, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, reinterpret_cast<void*>(base),
                   global_dims, global_strides, box_dims, elem_strides,
                   CU_TENSOR_MAP_INTERLEAVE_NONE, smem_swizzle,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(step1 comm w13 5d)");
  return out;
}

inline void UploadTensorMapLocal(const CUtensorMap& map, void** dev_ptr) {
  CheckCudaLocal(cudaGetLastError(), "UploadTensorMapLocal pending CUDA error");
  if (*dev_ptr == nullptr) {
    CheckCudaLocal(cudaMalloc(dev_ptr, sizeof(CUtensorMap)), "UploadTensorMapLocal cudaMalloc");
  }
  CheckCudaLocal(cudaMemcpy(*dev_ptr, &map, sizeof(CUtensorMap), cudaMemcpyHostToDevice),
                 "UploadTensorMapLocal cudaMemcpy");
}

struct CudaStageTimer {
  bool enabled = false;
  const char* tag = nullptr;
  int64_t t = 0;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;

  void Begin(bool enabled_in, const char* tag_in, int64_t t_in, cudaStream_t stream_in) {
    enabled = enabled_in;
    tag = tag_in;
    t = t_in;
    stream = stream_in;
    if (!enabled) return;
    CheckCudaLocal(cudaEventCreate(&start), "CudaStageTimer start create");
    CheckCudaLocal(cudaEventCreate(&end), "CudaStageTimer end create");
    CheckCudaLocal(cudaEventRecord(start, stream), "CudaStageTimer start record");
  }

  void End() {
    if (!enabled) return;
    CheckCudaLocal(cudaEventRecord(end, stream), "CudaStageTimer end record");
    CheckCudaLocal(cudaEventSynchronize(end), "CudaStageTimer end sync");
    float elapsed_ms = 0.0f;
    CheckCudaLocal(cudaEventElapsedTime(&elapsed_ms, start, end), "CudaStageTimer elapsed");
    const char* field = std::strcmp(tag, "moe_step1_timing") == 0 ? "step1" : "step2";
    std::fprintf(stderr, "[%s] impl=direct seq_len=%lld %s=%.3fms\n",
                 tag, static_cast<long long>(t), field, elapsed_ms);
    std::fflush(stderr);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
  }
};

inline int BuildActiveExpertList(const std::vector<int>& expert_counts_host,
                                 int* active_experts_dev,
                                 cudaStream_t stream) {
  int active_experts_host[kNumLocalExperts];
  int active_expert_count = 0;
  for (int le = 0; le < kNumLocalExperts; ++le) {
    if (expert_counts_host[le] > 0) {
      active_experts_host[active_expert_count++] = le;
    }
  }
  if (active_expert_count > 0) {
    CheckCudaLocal(cudaMemcpyAsync(active_experts_dev, active_experts_host,
                                   active_expert_count * sizeof(int),
                                   cudaMemcpyHostToDevice, stream),
                   "copy active experts");
  }
  return active_expert_count;
}

inline void UploadDirectTmaDescriptors(const TensorView& gemm1_weights,
                                       const TensorView& gemm2_weights,
                                       bool use_direct_tma,
                                       bool use_direct_tma_sw128) {
  void* gemm1_ptr = gemm1_weights.data_ptr();
  void* gemm2_ptr = gemm2_weights.data_ptr();
  if (g_cached_direct_gemm1_ptr == gemm1_ptr &&
      g_cached_direct_gemm2_ptr == gemm2_ptr &&
      g_cached_direct_tma == use_direct_tma &&
      g_cached_direct_tma_sw128 == use_direct_tma_sw128 &&
      (!use_direct_tma || !use_direct_tma_sw128 ||
       (g_step1_direct_w13_tma_desc_dev != nullptr &&
        g_step2_w2_tma_desc_dev != nullptr &&
        g_step2_w2_tma_desc_k4_dev != nullptr))) {
    return;
  }
  g_cached_direct_gemm1_ptr = gemm1_ptr;
  g_cached_direct_gemm2_ptr = gemm2_ptr;
  g_cached_direct_tma = use_direct_tma;
  g_cached_direct_tma_sw128 = use_direct_tma_sw128;

  const CUtensorMapSwizzle direct_smem_swizzle =
      use_direct_tma_sw128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE;
  CUtensorMap direct_w13_tmap =
      use_direct_tma_sw128
          ? EncodeTensorMap5DUint8Local(
                static_cast<uint8_t*>(gemm1_weights.data_ptr()),
                kBlock,
                kIntermediate,
                2,
                kHidden / kBlock,
                kNumLocalExperts,
                kHidden,
                static_cast<uint64_t>(kIntermediate) * static_cast<uint64_t>(kHidden),
                kBlock,
                static_cast<uint64_t>(2 * kIntermediate) * static_cast<uint64_t>(kHidden),
                kBlock,
                32,
                2,
                4,
                1,
                direct_smem_swizzle)
          : EncodeTensorMap5DUint8Local(
                static_cast<uint8_t*>(gemm1_weights.data_ptr()),
                kBlock,
                kHidden / kBlock,
                kIntermediate,
                2,
                kNumLocalExperts,
                kBlock,
                kHidden,
                static_cast<uint64_t>(kIntermediate) * static_cast<uint64_t>(kHidden),
                static_cast<uint64_t>(2 * kIntermediate) * static_cast<uint64_t>(kHidden),
                kBlock,
                8,
                32,
                2,
                1,
                direct_smem_swizzle);
  UploadTensorMapLocal(direct_w13_tmap, &g_step1_direct_w13_tma_desc_dev);

  if (!use_direct_tma || !use_direct_tma_sw128) {
    if (g_step2_w2_tma_desc_dev != nullptr) {
      CheckCudaLocal(cudaFree(g_step2_w2_tma_desc_dev), "free stale step2 w2 tma desc");
      g_step2_w2_tma_desc_dev = nullptr;
    }
    if (g_step2_w2_tma_desc_k4_dev != nullptr) {
      CheckCudaLocal(cudaFree(g_step2_w2_tma_desc_k4_dev), "free stale step2 w2 k4 tma desc");
      g_step2_w2_tma_desc_k4_dev = nullptr;
    }
    return;
  }
  CUtensorMap step2_w2_tmap = EncodeTensorMap4DUint8Local(
      static_cast<uint8_t*>(gemm2_weights.data_ptr()),
      kBlock,
      kHidden,
      kIntermediate / kBlock,
      kNumLocalExperts,
      kIntermediate,
      kBlock,
      static_cast<uint64_t>(kHidden) * static_cast<uint64_t>(kIntermediate),
      kBlock,
      64,
      8,
      1,
      CU_TENSOR_MAP_SWIZZLE_128B);
  UploadTensorMapLocal(step2_w2_tmap, &g_step2_w2_tma_desc_dev);

  CUtensorMap step2_w2_tmap_k4 = EncodeTensorMap4DUint8Local(
      static_cast<uint8_t*>(gemm2_weights.data_ptr()),
      kBlock,
      kHidden,
      kIntermediate / kBlock,
      kNumLocalExperts,
      kIntermediate,
      kBlock,
      static_cast<uint64_t>(kHidden) * static_cast<uint64_t>(kIntermediate),
      kBlock,
      64,
      4,
      1,
      CU_TENSOR_MAP_SWIZZLE_128B);
  UploadTensorMapLocal(step2_w2_tmap_k4, &g_step2_w2_tma_desc_k4_dev);
}

inline float bf16_to_float(uint16_t bits) {
  uint32_t u32 = static_cast<uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &u32, sizeof(out));
  return out;
}

inline float fp8_e4m3fn_to_float_host(uint8_t x) {
  const int sign = (x & 0x80) ? -1 : 1;
  const int exp = (x >> 3) & 0x0f;
  const int mant = x & 0x07;

  if (exp == 0) {
    if (mant == 0) return sign > 0 ? 0.0f : -0.0f;
    const float frac = static_cast<float>(mant) * 0.125f;
    return sign * std::ldexp(frac, -6);
  }

  if (exp == 0x0f && mant == 0x07) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  const float frac = 1.0f + static_cast<float>(mant) * 0.125f;
  return sign * std::ldexp(frac, exp - 7);
}

__device__ __forceinline__ float bf16_to_float_device(uint16_t bits) {
  uint32_t u32 = static_cast<uint32_t>(bits) << 16;
  return __uint_as_float(u32);
}

__device__ __forceinline__ uint16_t float_to_bf16_rne_device(float x) {
  uint32_t u32 = __float_as_uint(x);
  uint32_t lsb = (u32 >> 16) & 1u;
  uint32_t rounding_bias = 0x7fffu + lsb;
  return static_cast<uint16_t>((u32 + rounding_bias) >> 16);
}

__device__ __forceinline__ float sigmoidf_device(float x) {
  if (x >= 0.0f) {
    float z = expf(-x);
    return 1.0f / (1.0f + z);
  }
  float z = expf(x);
  return z / (1.0f + z);
}

struct TopKEntry {
  float val;
  int idx;
};

__device__ __forceinline__ void insert_topk8_device(TopKEntry* topk, float v, int idx) {
  if (v <= topk[7].val) return;
  topk[7] = {v, idx};
  for (int i = 7; i > 0 && topk[i].val > topk[i - 1].val; --i) {
    TopKEntry tmp = topk[i];
    topk[i] = topk[i - 1];
    topk[i - 1] = tmp;
  }
}

__global__ void routing_kernel(const float* __restrict__ logits, const uint16_t* __restrict__ bias_bf16,
                               int64_t t, int local_expert_offset, float routed_scaling_factor,
                               float* __restrict__ local_weight) {
  int tok = blockIdx.x * blockDim.x + threadIdx.x;
  if (tok >= t) return;

  const float* lrow = logits + static_cast<int64_t>(tok) * kNumExperts;
  float s[kNumExperts];
  float b[kNumExperts];
  for (int e = 0; e < kNumExperts; ++e) {
    s[e] = sigmoidf_device(lrow[e]);
    b[e] = bf16_to_float_device(bias_bf16[e]);
  }

  float group_scores[kNumGroups];
  for (int g = 0; g < kNumGroups; ++g) {
    float t1 = -INFINITY;
    float t2 = -INFINITY;
    int base = g * kGroupSize;
    for (int j = 0; j < kGroupSize; ++j) {
      int e = base + j;
      float v = s[e] + b[e];
      if (v > t1) {
        t2 = t1;
        t1 = v;
      } else if (v > t2) {
        t2 = v;
      }
    }
    group_scores[g] = t1 + t2;
  }

  TopKEntry grp[kTopKGroups] = {{-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1}};
  for (int g = 0; g < kNumGroups; ++g) {
    if (group_scores[g] > grp[3].val) {
      grp[3] = {group_scores[g], g};
      for (int i = 3; i > 0 && grp[i].val > grp[i - 1].val; --i) {
        TopKEntry tmp = grp[i];
        grp[i] = grp[i - 1];
        grp[i - 1] = tmp;
      }
    }
  }

  bool keep[kNumGroups] = {false, false, false, false, false, false, false, false};
  for (int i = 0; i < kTopKGroups; ++i) {
    if (grp[i].idx >= 0) keep[grp[i].idx] = true;
  }

  TopKEntry topk[kTopK] = {
      {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1},
      {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1},
  };
  for (int e = 0; e < kNumExperts; ++e) {
    int g = e / kGroupSize;
    if (!keep[g]) continue;
    float v = s[e] + b[e];
    insert_topk8_device(topk, v, e);
  }

  float wsum = 1e-20f;
  for (int i = 0; i < kTopK; ++i) {
    if (topk[i].idx >= 0) wsum += s[topk[i].idx];
  }

  float* lw_row = local_weight + static_cast<int64_t>(tok) * kNumLocalExperts;
  for (int i = 0; i < kTopK; ++i) {
    int ge = topk[i].idx;
    if (ge < 0) continue;
    int le = ge - local_expert_offset;
    if (0 <= le && le < kNumLocalExperts) {
      float w = (s[ge] / wsum) * routed_scaling_factor;
      lw_row[le] = w;
    }
  }
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ in, int64_t n, uint16_t* __restrict__ out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = float_to_bf16_rne_device(in[idx]);
}

// Routing-metadata builder kernels. Input is the existing dense
// local_weight[T, kNumLocalExperts] (0.0 for slots not routed to this rank's
// experts, nonzero otherwise). Output is a grouped-GEMM-friendly layout:
//   expert_counts[kNumLocalExperts]            — tokens per local expert
//   expert_offsets[kNumLocalExperts + 1]       — cumulative [0, sum]
//   permuted_token_ids[num_routed]             — original tok id per slot
//   permuted_weights[num_routed]               — per-(tok, expert) weight
// This layout is what the CUTLASS SM100 blockwise FP8 grouped GEMM on the
// contest's roadmap consumes directly.

__global__ void build_counts_kernel(const float* __restrict__ local_weight, int64_t t,
                                    int* __restrict__ expert_counts) {
  int tok = blockIdx.x * blockDim.x + threadIdx.x;
  if (tok >= t) return;
  const float* row = local_weight + static_cast<int64_t>(tok) * kNumLocalExperts;
  #pragma unroll
  for (int le = 0; le < kNumLocalExperts; ++le) {
    if (row[le] != 0.0f) atomicAdd(&expert_counts[le], 1);
  }
}

// Single-warp exclusive scan of 32 counts into 33 offsets. offsets[0]=0,
// offsets[32]=total routed. Launch <<<1, 32>>>.
__global__ void scan_offsets_kernel(const int* __restrict__ counts, int* __restrict__ offsets) {
  int tid = threadIdx.x;
  int v = counts[tid];
  #pragma unroll
  for (int k = 1; k < 32; k *= 2) {
    int n = __shfl_up_sync(0xffffffffu, v, k);
    if (tid >= k) v += n;
  }
  int incl = v;
  int excl = incl - counts[tid];
  offsets[tid] = excl;
  if (tid == 31) offsets[32] = incl;
}

__global__ void scatter_placements_kernel(const float* __restrict__ local_weight, int64_t t,
                                          const int* __restrict__ expert_offsets,
                                          int* __restrict__ running_counter,
                                          int* __restrict__ permuted_token_ids,
                                          float* __restrict__ permuted_weights) {
  int tok = blockIdx.x * blockDim.x + threadIdx.x;
  if (tok >= t) return;
  const float* row = local_weight + static_cast<int64_t>(tok) * kNumLocalExperts;
  #pragma unroll
  for (int le = 0; le < kNumLocalExperts; ++le) {
    float w = row[le];
    if (w == 0.0f) continue;
    int slot = atomicAdd(&running_counter[le], 1);
    int pos = expert_offsets[le] + slot;
    permuted_token_ids[pos] = tok;
    permuted_weights[pos] = w;
  }
}

void write_binary_file(const std::filesystem::path& path, const void* data, size_t bytes) {
  FILE* fp = std::fopen(path.c_str(), "wb");
  if (fp == nullptr) {
    return;
  }
  if (bytes > 0) {
    std::fwrite(data, 1, bytes, fp);
  }
  std::fclose(fp);
}

void maybe_dump_step1_debug(const char* dump_dir, int64_t t, int total_routed,
                            const std::vector<int>& expert_counts_host,
                            const std::vector<int>& expert_offsets_host,
                            const int* permuted_token_ids_dev,
                            const float* permuted_weights_dev,
                            int debug_output_mode,
                            const float* c_perm_all_dev,
                            cudaStream_t stream) {
  if (dump_dir == nullptr || dump_dir[0] == '\0' || total_routed <= 0) {
    return;
  }

  cudaError_t st = cudaStreamSynchronize(stream);
  if (st != cudaSuccess) {
    std::fprintf(stderr, "[step1_dump] sync failed: %s\n", cudaGetErrorString(st));
    return;
  }

  std::filesystem::path dir(dump_dir);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    std::fprintf(stderr, "[step1_dump] mkdir failed for %s: %s\n", dir.c_str(), ec.message().c_str());
    return;
  }

  std::vector<int> permuted_token_ids(total_routed);
  std::vector<float> permuted_weights(total_routed);
  std::vector<float> c_perm(static_cast<size_t>(total_routed) * kIntermediate);

  cudaMemcpy(permuted_token_ids.data(), permuted_token_ids_dev,
             static_cast<size_t>(total_routed) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(permuted_weights.data(), permuted_weights_dev,
             static_cast<size_t>(total_routed) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(c_perm.data(), c_perm_all_dev,
             static_cast<size_t>(total_routed) * kIntermediate * sizeof(float),
             cudaMemcpyDeviceToHost);

  write_binary_file(dir / "expert_counts.bin", expert_counts_host.data(),
                    expert_counts_host.size() * sizeof(int));
  write_binary_file(dir / "expert_offsets.bin", expert_offsets_host.data(),
                    expert_offsets_host.size() * sizeof(int));
  write_binary_file(dir / "permuted_token_ids.bin", permuted_token_ids.data(),
                    permuted_token_ids.size() * sizeof(int));
  write_binary_file(dir / "permuted_weights.bin", permuted_weights.data(),
                    permuted_weights.size() * sizeof(float));
  write_binary_file(dir / "c_perm.bin", c_perm.data(), c_perm.size() * sizeof(float));

  FILE* meta = std::fopen((dir / "meta.txt").c_str(), "w");
  if (meta != nullptr) {
    std::fprintf(meta, "t=%lld\n", static_cast<long long>(t));
    std::fprintf(meta, "total_routed=%d\n", total_routed);
    std::fprintf(meta, "intermediate=%d\n", kIntermediate);
    std::fprintf(meta, "debug_output_mode=%d\n", debug_output_mode);
    std::fclose(meta);
  }
}

// Persistent per-process workspace: all transient device buffers grow on
// demand and are reused across calls. The per-call cudaMalloc/cudaFree cost
// was measurable on small-T workloads (observed ~61ms regression on T=1 with
// grouped path on B200); this eliminates it. Buffers leak at process exit —
// acceptable for a per-process benchmark harness.
struct MoeWorkspace {
  int64_t cap_t = 0;
  int64_t cap_t_grouped = 0;
  int64_t cap_routed = 0;
  float* local_weight_dev = nullptr;
  float* out_acc_dev = nullptr;
  int* expert_counts_dev = nullptr;
  int* expert_offsets_dev = nullptr;
  int* running_counter_dev = nullptr;
  int* permuted_token_ids_dev = nullptr;
  int* active_experts_dev = nullptr;
  float* permuted_weights_dev = nullptr;
  float* c_perm_all_dev = nullptr;
  uint8_t* c_perm_q_dev = nullptr;
  float* c_perm_scale_dev = nullptr;

  void ensure_core(int64_t t) {
    if (t <= cap_t && local_weight_dev != nullptr) return;
    if (local_weight_dev) cudaFree(local_weight_dev);
    if (out_acc_dev) cudaFree(out_acc_dev);
    cap_t = t;
    cudaMalloc(&local_weight_dev, static_cast<size_t>(t) * kNumLocalExperts * sizeof(float));
    cudaMalloc(&out_acc_dev, static_cast<size_t>(t) * kHidden * sizeof(float));
  }

  void ensure_grouped(int64_t t) {
    if (t <= cap_t_grouped && permuted_token_ids_dev != nullptr) return;
    if (expert_counts_dev) cudaFree(expert_counts_dev);
    if (expert_offsets_dev) cudaFree(expert_offsets_dev);
    if (running_counter_dev) cudaFree(running_counter_dev);
    if (permuted_token_ids_dev) cudaFree(permuted_token_ids_dev);
    if (active_experts_dev) cudaFree(active_experts_dev);
    if (permuted_weights_dev) cudaFree(permuted_weights_dev);
    cap_t_grouped = t;
    const int64_t max_routed = t * kTopK;
    cudaMalloc(&expert_counts_dev, kNumLocalExperts * sizeof(int));
    cudaMalloc(&expert_offsets_dev, (kNumLocalExperts + 1) * sizeof(int));
    cudaMalloc(&running_counter_dev, kNumLocalExperts * sizeof(int));
    cudaMalloc(&permuted_token_ids_dev, static_cast<size_t>(max_routed) * sizeof(int));
    cudaMalloc(&active_experts_dev, kNumLocalExperts * sizeof(int));
    cudaMalloc(&permuted_weights_dev, static_cast<size_t>(max_routed) * sizeof(float));
  }

  void ensure_routed_step1(int64_t routed_rows) {
    if (routed_rows <= cap_routed && c_perm_all_dev != nullptr &&
        c_perm_q_dev != nullptr && c_perm_scale_dev != nullptr) {
      return;
    }
    if (c_perm_all_dev) cudaFree(c_perm_all_dev);
    if (c_perm_q_dev) cudaFree(c_perm_q_dev);
    if (c_perm_scale_dev) cudaFree(c_perm_scale_dev);
    cap_routed = routed_rows;
    if (routed_rows > 0) {
      cudaMalloc(&c_perm_all_dev,
                 static_cast<size_t>(routed_rows) * kIntermediate * sizeof(float));
      cudaMalloc(&c_perm_q_dev,
                 static_cast<size_t>(routed_rows) * kIntermediate * sizeof(uint8_t));
      cudaMalloc(&c_perm_scale_dev,
                 static_cast<size_t>(routed_rows) * (kIntermediate / kBlock) * 4 * sizeof(float));
    }
  }
};

}  // namespace

void moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048_impl(
    TensorView routing_logits, TensorView routing_bias, TensorView hidden_states,
    TensorView hidden_states_scale, TensorView gemm1_weights, TensorView gemm1_weights_scale,
    TensorView gemm2_weights, TensorView gemm2_weights_scale, int64_t local_expert_offset,
    double routed_scaling_factor, TensorView output) {
  TVM_FFI_ICHECK_EQ(routing_logits.ndim(), 2);
  TVM_FFI_ICHECK_EQ(routing_bias.ndim(), 1);
  TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2);
  TVM_FFI_ICHECK_EQ(hidden_states_scale.ndim(), 2);
  TVM_FFI_ICHECK_EQ(gemm1_weights.ndim(), 3);
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.ndim(), 3);
  TVM_FFI_ICHECK_EQ(gemm2_weights.ndim(), 3);
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.ndim(), 3);
  TVM_FFI_ICHECK_EQ(output.ndim(), 2);

  int64_t t = routing_logits.size(0);
  TVM_FFI_ICHECK_EQ(routing_logits.size(1), kNumExperts);
  TVM_FFI_ICHECK_EQ(routing_bias.size(0), kNumExperts);
  TVM_FFI_ICHECK_EQ(hidden_states.size(0), t);
  TVM_FFI_ICHECK_EQ(hidden_states.size(1), kHidden);
  TVM_FFI_ICHECK_EQ(hidden_states_scale.size(0), kHidden / kBlock);
  TVM_FFI_ICHECK_EQ(hidden_states_scale.size(1), t);
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(0), kNumLocalExperts);
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(1), 2 * kIntermediate);
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(2), kHidden);
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(0), kNumLocalExperts);
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(1), (2 * kIntermediate) / kBlock);
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(2), kHidden / kBlock);
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(0), kNumLocalExperts);
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(1), kHidden);
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(2), kIntermediate);
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(0), kNumLocalExperts);
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(1), kHidden / kBlock);
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(2), kIntermediate / kBlock);
  TVM_FFI_ICHECK_EQ(output.size(0), t);
  TVM_FFI_ICHECK_EQ(output.size(1), kHidden);

  DLDevice dev = output.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  // Per-stage stream syncs + stderr line are diagnostic only. They add ~4 full
  // device round-trips per MoE call, which hides real kernel cost in benchmark
  // runs. Opt in with FIB_MOE_PROFILE=1 (any non-empty value) when A/B testing.
  const bool kProfile = std::getenv("FIB_MOE_PROFILE") != nullptr;

  static reference_backend::ReferenceGemmModule gemm_mod(kHidden, kIntermediate, kBlock);

  const auto t0_total =
      kProfile ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

  static MoeWorkspace ws;
  ws.ensure_core(t);
  float* local_weight_dev = ws.local_weight_dev;
  float* out_acc_dev = ws.out_acc_dev;

  cudaMemsetAsync(local_weight_dev, 0, static_cast<size_t>(t) * kNumLocalExperts * sizeof(float), stream);
  cudaMemsetAsync(out_acc_dev, 0, static_cast<size_t>(t) * kHidden * sizeof(float), stream);

  ws.ensure_grouped(t);
  int* expert_counts_dev = ws.expert_counts_dev;
  int* expert_offsets_dev = ws.expert_offsets_dev;
  int* running_counter_dev = ws.running_counter_dev;
  int* permuted_token_ids_dev = ws.permuted_token_ids_dev;
  int* active_experts_dev = ws.active_experts_dev;
  float* permuted_weights_dev = ws.permuted_weights_dev;
  cudaMemsetAsync(expert_counts_dev, 0, kNumLocalExperts * sizeof(int), stream);
  cudaMemsetAsync(running_counter_dev, 0, kNumLocalExperts * sizeof(int), stream);
  const bool use_direct_impl = UseDirectImplementation();
  const bool need_step1_debug_dump = std::getenv("FIB_DEBUG_DUMP_STEP1_DIR") != nullptr;

  std::chrono::high_resolution_clock::time_point t0_routing, t1_routing;
  std::chrono::high_resolution_clock::time_point t0_dequant, t1_dequant;
  std::chrono::high_resolution_clock::time_point t0_expert, t1_expert;
  std::chrono::high_resolution_clock::time_point t0_out, t1_out;

  if (kProfile) t0_routing = std::chrono::high_resolution_clock::now();
  constexpr int kRoutingThreads = 128;
  int routing_blocks = static_cast<int>((t + kRoutingThreads - 1) / kRoutingThreads);
  routing_kernel<<<routing_blocks, kRoutingThreads, 0, stream>>>(
      static_cast<const float*>(routing_logits.data_ptr()),
      static_cast<const uint16_t*>(routing_bias.data_ptr()), t, static_cast<int>(local_expert_offset),
      static_cast<float>(routed_scaling_factor), local_weight_dev);
  if (kProfile) {
    cudaStreamSynchronize(stream);
    t1_routing = std::chrono::high_resolution_clock::now();
  }

  int64_t n_hidden = t * kHidden;
  if (kProfile) {
    t0_dequant = std::chrono::high_resolution_clock::now();
    // FP8 activation dequant is fused inside GEMM1 path now.
    t1_dequant = t0_dequant;
  }

  if (kProfile) t0_expert = std::chrono::high_resolution_clock::now();

  std::vector<int> expert_counts_host(kNumLocalExperts, 0);
  std::vector<int> expert_offsets_host(kNumLocalExperts + 1, 0);

  // Build grouped-GEMM metadata.
  constexpr int kMetaThreads = 128;
  int meta_blocks = static_cast<int>((t + kMetaThreads - 1) / kMetaThreads);
  build_counts_kernel<<<meta_blocks, kMetaThreads, 0, stream>>>(local_weight_dev, t,
                                                                expert_counts_dev);
  scan_offsets_kernel<<<1, 32, 0, stream>>>(expert_counts_dev, expert_offsets_dev);
  scatter_placements_kernel<<<meta_blocks, kMetaThreads, 0, stream>>>(
      local_weight_dev, t, expert_offsets_dev, running_counter_dev, permuted_token_ids_dev,
      permuted_weights_dev);

  const bool use_device_metadata_direct =
      use_direct_impl && !need_step1_debug_dump && t <= 128;
  const bool need_metadata_host = !use_device_metadata_direct;
  if (need_metadata_host) {
    cudaMemcpyAsync(expert_counts_host.data(), expert_counts_dev,
                    kNumLocalExperts * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(expert_offsets_host.data(), expert_offsets_dev,
                    (kNumLocalExperts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaError_t meta_sync = cudaStreamSynchronize(stream);
    TVM_FFI_ICHECK_EQ(meta_sync, cudaSuccess)
        << "grouped-metadata sync failed before backend launch: "
        << cudaGetErrorString(meta_sync);
  }

  const int total_routed =
      need_metadata_host ? expert_offsets_host[kNumLocalExperts]
                         : static_cast<int>(t * kTopK);
  const int active_expert_count =
      need_metadata_host ? BuildActiveExpertList(expert_counts_host, active_experts_dev, stream) : 0;
  if (use_direct_impl && total_routed > 0) {
    const bool kUseDirectTma = std::getenv("FIB_MOE_COMM_USE_TMA") != nullptr;
    const bool kUseDirectTmaSw128 = std::getenv("FIB_MOE_DIRECT_TMA_SW128") != nullptr;
    const bool kUseDirectBSw128 = std::getenv("FIB_MOE_DIRECT_B_SW128") != nullptr;
    const int step2_pipeline_variant = ParseStep2PipelineVariant();
    UploadDirectTmaDescriptors(gemm1_weights, gemm2_weights, kUseDirectTma, kUseDirectTmaSw128);

    const bool kMeasureStep1CommOnly = std::getenv("FIB_MEASURE_STEP1_COMM_ONLY") != nullptr;
    if (kMeasureStep1CommOnly) {
      const bool kUseCommTma = std::getenv("FIB_MOE_COMM_USE_TMA") != nullptr;
      int comm_tma_mode = 0;
      int comm_out_rows = kBlock;
      int comm_h_tiles = kBlock / 32;
      bool comm_double_buffer = std::getenv("FIB_MOE_COMM_DOUBLE_BUFFER") != nullptr;
      bool comm_skip_hidden = std::getenv("FIB_MOE_COMM_SKIP_HIDDEN") != nullptr;
      bool comm_skip_weight = std::getenv("FIB_MOE_COMM_SKIP_WEIGHT") != nullptr;
      if (const char* comm_out_rows_env = std::getenv("FIB_MOE_COMM_OUT_ROWS")) {
        const int parsed = std::atoi(comm_out_rows_env);
        if (parsed == 16 || parsed == 32 || parsed == 64 || parsed == 128) {
          comm_out_rows = parsed;
        }
      }
      if (const char* comm_h_tiles_env = std::getenv("FIB_MOE_COMM_H_TILES")) {
        const int parsed = std::atoi(comm_h_tiles_env);
        if (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8 || parsed == 16) {
          comm_h_tiles = parsed;
        }
      }
      if (const char* comm_tma_mode_env = std::getenv("FIB_MOE_COMM_TMA_MODE")) {
        if (std::strcmp(comm_tma_mode_env, "raw7") == 0) {
          comm_tma_mode = 1;
        } else if (std::strcmp(comm_tma_mode_env, "raw14") == 0) {
          comm_tma_mode = 2;
        } else if (std::strcmp(comm_tma_mode_env, "half") == 0) {
          comm_tma_mode = 3;
        }
      }
      if (kUseCommTma) {
        CUtensorMap comm_w13_tmap{};
        const CUtensorMapSwizzle comm_smem_swizzle =
            std::getenv("FIB_MOE_COMM_TMA_SW128") != nullptr
                ? CU_TENSOR_MAP_SWIZZLE_128B
                : CU_TENSOR_MAP_SWIZZLE_NONE;
        if (comm_tma_mode == 0) {
          const int comm_gate_up_tiles = 2;
          comm_w13_tmap = EncodeTensorMap5DUint8Local(
              static_cast<uint8_t*>(gemm1_weights.data_ptr()),
              kBlock,
              kHidden / kBlock,
              kIntermediate,
              comm_gate_up_tiles,
              kNumLocalExperts,
              kBlock,
              kHidden,
              static_cast<uint64_t>(kIntermediate) * static_cast<uint64_t>(kHidden),
              static_cast<uint64_t>(2 * kIntermediate) * static_cast<uint64_t>(kHidden),
              kBlock,
              comm_h_tiles,
              comm_out_rows,
              comm_gate_up_tiles,
              1,
              comm_smem_swizzle);
        } else if (comm_tma_mode == 3) {
          const int comm_gate_up_tiles = 1;
          comm_w13_tmap = EncodeTensorMap5DUint8Local(
              static_cast<uint8_t*>(gemm1_weights.data_ptr()),
              kBlock,
              kHidden / kBlock,
              kIntermediate,
              comm_gate_up_tiles,
              kNumLocalExperts,
              kBlock,
              kHidden,
              static_cast<uint64_t>(kIntermediate) * static_cast<uint64_t>(kHidden),
              static_cast<uint64_t>(2 * kIntermediate) * static_cast<uint64_t>(kHidden),
              kBlock,
              comm_h_tiles,
              comm_out_rows,
              comm_gate_up_tiles,
              1,
              comm_smem_swizzle);
        } else {
          const uint64_t total_bytes =
              static_cast<uint64_t>(kNumLocalExperts) *
              static_cast<uint64_t>(2 * kIntermediate) *
              static_cast<uint64_t>(kHidden);
          const uint64_t total_planes =
              total_bytes / (static_cast<uint64_t>(kBlock) * static_cast<uint64_t>(kBlock));
          comm_w13_tmap = EncodeTensorMap3DUint8Local(
              static_cast<uint8_t*>(gemm1_weights.data_ptr()),
              kBlock,
              kBlock,
              total_planes,
              kBlock,
              static_cast<uint64_t>(kBlock) * static_cast<uint64_t>(kBlock),
              kBlock,
              kBlock,
              8,
              comm_smem_swizzle);
        }
        UploadTensorMapLocal(comm_w13_tmap, &g_step1_comm_w13_tma_desc_dev);
      }

      cudaEvent_t comm_start = nullptr;
      cudaEvent_t comm_end = nullptr;
      cudaEventCreate(&comm_start);
      cudaEventCreate(&comm_end);
      cudaEventRecord(comm_start, stream);
      cudaError_t st_comm = direct_backend::RunStep1CommOnlyDirect(
          static_cast<const uint8_t*>(hidden_states.data_ptr()),
          t,
          expert_counts_dev,
          expert_offsets_dev,
          permuted_token_ids_dev,
          static_cast<const uint8_t*>(gemm1_weights.data_ptr()),
          kUseCommTma ? g_step1_comm_w13_tma_desc_dev : nullptr,
          comm_tma_mode,
          kBlock,
          comm_out_rows,
          comm_h_tiles,
          comm_double_buffer,
          comm_skip_hidden,
          comm_skip_weight,
          stream);
      TVM_FFI_ICHECK_EQ(st_comm, cudaSuccess)
          << "direct Step1(comm-only) launch failed: " << cudaGetErrorString(st_comm);
      cudaEventRecord(comm_end, stream);
      cudaEventSynchronize(comm_end);
      float comm_ms = 0.0f;
      cudaEventElapsedTime(&comm_ms, comm_start, comm_end);
      std::fprintf(stderr,
                   "[moe_step1_comm_timing] impl=direct seq_len=%lld step1_comm_only=%.3fms\n",
                   static_cast<long long>(t), comm_ms);
      std::fflush(stderr);
      cudaEventDestroy(comm_start);
      cudaEventDestroy(comm_end);
    }

    CudaStageTimer step1_timer;
    step1_timer.Begin(kProfile, "moe_step1_timing", t, stream);

    // GEMM1 emits one compact intermediate row per routed (token, local-expert)
    // pair. The flat buffer is therefore [sum_e expert_counts[e], I], where
    // each expert owns a contiguous slice determined by expert_offsets[e].
    ws.ensure_routed_step1(total_routed);
    int step1_debug_output_mode = 0;
    if (const char* debug_raw_env = std::getenv("FIB_DEBUG_STEP1_OUTPUT")) {
      if (std::strcmp(debug_raw_env, "gate") == 0) {
        step1_debug_output_mode = 1;
      } else if (std::strcmp(debug_raw_env, "up") == 0) {
        step1_debug_output_mode = 2;
      } else if (std::strcmp(debug_raw_env, "gate_unscaled") == 0) {
        step1_debug_output_mode = 3;
      } else if (std::strcmp(debug_raw_env, "up_unscaled") == 0) {
        step1_debug_output_mode = 4;
      }
    }
    if (std::getenv("FIB_STEP1_ABLATE_ACCUM56") != nullptr) {
      step1_debug_output_mode = 99;
    }
    if (std::getenv("FIB_STEP1_ABLATE_ACCUM56_RAW_GATE") != nullptr) {
      step1_debug_output_mode = 98;
    }
    int step1_tcgen_accum_mode = 2;
    if (const char* accum_env = std::getenv("FIB_TCGEN_ACCUM")) {
      if (std::strcmp(accum_env, "f16") == 0 || std::strcmp(accum_env, "fp16") == 0) {
        step1_tcgen_accum_mode = 1;
      } else if (std::strcmp(accum_env, "tf32_scale") == 0 ||
                 std::strcmp(accum_env, "tmem_scale") == 0) {
        step1_tcgen_accum_mode = 2;
      } else if (std::strcmp(accum_env, "tf32_scale_garbage") == 0 ||
                 std::strcmp(accum_env, "tmem_scale_garbage") == 0) {
        step1_tcgen_accum_mode = 3;
      }
    }
    cudaError_t st1 = direct_backend::RunStep1AllExpertsDirect(
        static_cast<const uint8_t*>(hidden_states.data_ptr()),
        static_cast<const float*>(hidden_states_scale.data_ptr()),
        t,
        expert_counts_dev,
        expert_offsets_dev,
        permuted_token_ids_dev,
        need_metadata_host ? active_experts_dev : nullptr,
        need_metadata_host ? active_expert_count : 0,
        static_cast<const uint8_t*>(gemm1_weights.data_ptr()),
        static_cast<const float*>(gemm1_weights_scale.data_ptr()),
        kUseDirectTma ? g_step1_direct_w13_tma_desc_dev : nullptr,
        step1_debug_output_mode,
        ws.c_perm_all_dev,
        stream,
        kUseDirectTma && kUseDirectTmaSw128,
        kUseDirectBSw128,
        step1_tcgen_accum_mode);
    TVM_FFI_ICHECK_EQ(st1, cudaSuccess)
        << "direct Step1(all experts) launch failed: " << cudaGetErrorString(st1);
    step1_timer.End();

    if (need_step1_debug_dump) {
      maybe_dump_step1_debug(
          std::getenv("FIB_DEBUG_DUMP_STEP1_DIR"),
          t,
          total_routed,
          expert_counts_host,
          expert_offsets_host,
          permuted_token_ids_dev,
          permuted_weights_dev,
          step1_debug_output_mode,
          ws.c_perm_all_dev,
          stream);
    }

    CudaStageTimer step2_timer;
    step2_timer.Begin(kProfile, "moe_step2_timing", t, stream);
    cudaError_t st2_pre = direct_backend::LaunchStep2PrequantCperm(
        ws.c_perm_all_dev, total_routed, ws.c_perm_q_dev, ws.c_perm_scale_dev, stream);
    TVM_FFI_ICHECK_EQ(st2_pre, cudaSuccess)
        << "direct Step2(prequant c_perm) launch failed: " << cudaGetErrorString(st2_pre);
    const bool step2_k4_variant =
        step2_pipeline_variant == direct_backend::kStep2PipeM64N8K4DbBatch2 ||
        step2_pipeline_variant == direct_backend::kStep2PipeM64N16K4DbBatch2;
    const void* step2_tma_desc =
        kUseDirectTma ? (step2_k4_variant ? g_step2_w2_tma_desc_k4_dev : g_step2_w2_tma_desc_dev)
                      : nullptr;
    cudaError_t st2 = direct_backend::LaunchStep2DirectAllExperts(
        expert_counts_dev, expert_offsets_dev, permuted_token_ids_dev,
        permuted_weights_dev,
        need_metadata_host ? active_experts_dev : nullptr,
        need_metadata_host ? active_expert_count : 0,
        ws.c_perm_q_dev, ws.c_perm_scale_dev, static_cast<const uint8_t*>(gemm2_weights.data_ptr()),
        static_cast<const float*>(gemm2_weights_scale.data_ptr()),
        step2_tma_desc, out_acc_dev, stream, step2_pipeline_variant);
    TVM_FFI_ICHECK_EQ(st2, cudaSuccess)
        << "direct Step2(all experts) launch failed: " << cudaGetErrorString(st2);
    step2_timer.End();
  } else {
    gemm_mod.EnsureWorkspace(t, stream);
    for (int le = 0; le < kNumLocalExperts; ++le) {
      int n_rows = expert_counts_host[le];
      if (n_rows == 0) continue;
      int start = expert_offsets_host[le];
      gemm_mod.RunExpertPermuted(
          static_cast<const uint8_t*>(hidden_states.data_ptr()),
          static_cast<const float*>(hidden_states_scale.data_ptr()), t, n_rows,
          permuted_token_ids_dev + start, permuted_weights_dev + start, le,
          static_cast<const uint8_t*>(gemm1_weights.data_ptr()),
          static_cast<const float*>(gemm1_weights_scale.data_ptr()),
          static_cast<const uint8_t*>(gemm2_weights.data_ptr()),
          static_cast<const float*>(gemm2_weights_scale.data_ptr()), out_acc_dev, stream);
    }
  }
  if (kProfile) {
    cudaStreamSynchronize(stream);
    t1_expert = std::chrono::high_resolution_clock::now();
  }

  if (kProfile) t0_out = std::chrono::high_resolution_clock::now();
  constexpr int kOutThreads = 256;
  int out_blocks = static_cast<int>((n_hidden + kOutThreads - 1) / kOutThreads);
  f32_to_bf16_kernel<<<out_blocks, kOutThreads, 0, stream>>>(
      out_acc_dev, n_hidden, static_cast<uint16_t*>(output.data_ptr()));
  // The direct path reuses raw static workspace and global TMA descriptors across
  // invocations. Complete the stream before returning so the next workload cannot
  // overwrite those resources while kernels from this call are still reading them.
  cudaError_t final_sync = cudaStreamSynchronize(stream);
  TVM_FFI_ICHECK_EQ(final_sync, cudaSuccess)
      << "direct MoE final stream sync failed: " << cudaGetErrorString(final_sync);
  if (kProfile) {
    t1_out = std::chrono::high_resolution_clock::now();
    const auto t1_total = t1_out;
    auto ms = [](const auto& a, const auto& b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr,
                 "[moe_step_timing] seq_len=%lld routing=%.3fms dequant=%.3fms expert=%.3fms "
                 "out=%.3fms total=%.3fms\n",
                 static_cast<long long>(t), ms(t0_routing, t1_routing), ms(t0_dequant, t1_dequant),
                 ms(t0_expert, t1_expert), ms(t0_out, t1_out), ms(t0_total, t1_total));
    std::fflush(stderr);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048,
                              moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048_impl);
