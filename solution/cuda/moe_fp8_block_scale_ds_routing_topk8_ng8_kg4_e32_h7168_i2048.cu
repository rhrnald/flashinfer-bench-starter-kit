#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "mxfp_gemm_module.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

using tvm::ffi::TensorView;

namespace {

constexpr int kNumExperts = 256;
constexpr int kNumLocalExperts = 32;
constexpr int kHidden = 7168;
constexpr int kIntermediate = 2048;
constexpr int kBlock = 128;
constexpr int kNumGroups = 8;
constexpr int kGroupSize = 32;
constexpr int kTopKGroups = 4;
constexpr int kTopK = 8;

inline float bf16_to_float(uint16_t bits) {
  uint32_t u32 = static_cast<uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &u32, sizeof(out));
  return out;
}

__device__ __forceinline__ float bf16_to_float_device(uint16_t bits) {
  uint32_t u32 = static_cast<uint32_t>(bits) << 16;
  return __uint_as_float(u32);
}

__device__ __forceinline__ float fp8_e4m3fn_to_float_device(uint8_t x) {
  int sign = (x & 0x80) ? -1 : 1;
  int exp = (x >> 3) & 0x0f;
  int mant = x & 0x07;

  if (exp == 0) {
    if (mant == 0) return sign == 1 ? 0.0f : -0.0f;
    float frac = static_cast<float>(mant) * 0.125f;
    return sign * ldexpf(frac, -6);
  }

  float frac = 1.0f + static_cast<float>(mant) * 0.125f;
  return sign * ldexpf(frac, exp - 7);
}

__constant__ float kFp8DecodeTable[256];

inline float fp8_e4m3fn_to_float_table_host(uint8_t x) {
  int sign = (x & 0x80) ? -1 : 1;
  int exp = (x >> 3) & 0x0f;
  int mant = x & 0x07;
  if (exp == 0) {
    if (mant == 0) return sign == 1 ? 0.0f : -0.0f;
    float frac = static_cast<float>(mant) * 0.125f;
    return sign * std::ldexp(frac, -6);
  }
  float frac = 1.0f + static_cast<float>(mant) * 0.125f;
  return sign * std::ldexp(frac, exp - 7);
}

void initialize_fp8_decode_table_once() {
  static bool initialized = false;
  if (initialized) return;
  float table[256];
  for (int i = 0; i < 256; ++i) {
    table[i] = fp8_e4m3fn_to_float_table_host(static_cast<uint8_t>(i));
  }
  cudaMemcpyToSymbol(kFp8DecodeTable, table, sizeof(table));
  initialized = true;
}

__device__ __forceinline__ float fp8_e4m3fn_to_float_table_device(uint8_t x) {
  return kFp8DecodeTable[x];
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
                               float* __restrict__ local_weight, uint8_t* __restrict__ expert_used) {
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
      expert_used[le] = 1;
    }
  }
}

__global__ void dequant_hidden_kernel(const uint8_t* __restrict__ hidden_fp8,
                                      const float* __restrict__ hidden_scale, int64_t t,
                                      float* __restrict__ a_out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = t * kHidden;
  if (idx >= total) return;

  int64_t tok = idx / kHidden;
  int h = static_cast<int>(idx - tok * kHidden);
  int hb = h / kBlock;
  float scale = hidden_scale[static_cast<int64_t>(hb) * t + tok];
  float v = fp8_e4m3fn_to_float_device(hidden_fp8[idx]);
  a_out[idx] = v * scale;
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
                                          float* __restrict__ permuted_weights,
                                          int* __restrict__ permuted_experts) {
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
    permuted_experts[pos] = le;
  }
}

__global__ void all_experts_step1_swiglu_kernel(
    const float* __restrict__ a, const int* __restrict__ expert_offsets,
    const int* __restrict__ permuted_token_ids, const int* __restrict__ permuted_experts,
    const uint8_t* __restrict__ w13, const float* __restrict__ s13,
    int64_t max_routed, float* __restrict__ c_all) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int total_routed = expert_offsets[kNumLocalExperts];
  int64_t total = static_cast<int64_t>(total_routed) * kIntermediate;
  if (idx >= total || idx >= max_routed * kIntermediate) return;

  int pos = static_cast<int>(idx / kIntermediate);
  int i = static_cast<int>(idx - static_cast<int64_t>(pos) * kIntermediate);
  int tok = permuted_token_ids[pos];
  int le = permuted_experts[pos];

  const float* a_row = a + static_cast<int64_t>(tok) * kHidden;
  const uint8_t* w13_e = w13 + static_cast<int64_t>(le) * (2 * kIntermediate) * kHidden;
  const float* s13_e = s13 + static_cast<int64_t>(le) * ((2 * kIntermediate) / kBlock) *
                                 (kHidden / kBlock);
  int ib = i / kBlock;
  int j1 = i;
  int j2 = i + kIntermediate;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  #pragma unroll
  for (int hb = 0; hb < kHidden / kBlock; ++hb) {
    float scale1 = s13_e[ib * (kHidden / kBlock) + hb];
    float scale2 = s13_e[(ib + kIntermediate / kBlock) * (kHidden / kBlock) + hb];
    int h0 = hb * kBlock;
    float block_raw1 = 0.0f;
    float block_raw2 = 0.0f;
    #pragma unroll
    for (int u = 0; u < kBlock; ++u) {
      int h = h0 + u;
      float av = a_row[h];
      float w1 = fp8_e4m3fn_to_float_table_device(w13_e[static_cast<int64_t>(j1) * kHidden + h]);
      float w2 = fp8_e4m3fn_to_float_table_device(w13_e[static_cast<int64_t>(j2) * kHidden + h]);
      block_raw1 += av * w1;
      block_raw2 += av * w2;
    }
    acc1 += block_raw1 * scale1;
    acc2 += block_raw2 * scale2;
  }
  c_all[static_cast<int64_t>(pos) * kIntermediate + i] =
      acc1 * (acc2 / (1.0f + expf(-acc2)));
}

__global__ void all_experts_step2_scatter_kernel(
    const float* __restrict__ c_all, const int* __restrict__ expert_offsets,
    const int* __restrict__ permuted_token_ids, const float* __restrict__ permuted_weights,
    const int* __restrict__ permuted_experts, const uint8_t* __restrict__ w2,
    const float* __restrict__ s2, int64_t max_routed, float* __restrict__ out_acc) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int total_routed = expert_offsets[kNumLocalExperts];
  int64_t total = static_cast<int64_t>(total_routed) * kHidden;
  if (idx >= total || idx >= max_routed * kHidden) return;

  int pos = static_cast<int>(idx / kHidden);
  int h = static_cast<int>(idx - static_cast<int64_t>(pos) * kHidden);
  int tok = permuted_token_ids[pos];
  int le = permuted_experts[pos];
  float route_w = permuted_weights[pos];

  const float* c_row = c_all + static_cast<int64_t>(pos) * kIntermediate;
  const uint8_t* w2_e = w2 + static_cast<int64_t>(le) * kHidden * kIntermediate;
  const float* s2_e = s2 + static_cast<int64_t>(le) * (kHidden / kBlock) *
                               (kIntermediate / kBlock);
  int hb = h / kBlock;
  float acc = 0.0f;
  #pragma unroll
  for (int ib = 0; ib < kIntermediate / kBlock; ++ib) {
    float scale = s2_e[hb * (kIntermediate / kBlock) + ib];
    int i0 = ib * kBlock;
    float block_raw = 0.0f;
    #pragma unroll
    for (int u = 0; u < kBlock; ++u) {
      int i = i0 + u;
      float cv = c_row[i];
      float wv = fp8_e4m3fn_to_float_table_device(w2_e[static_cast<int64_t>(h) * kIntermediate + i]);
      block_raw += cv * wv;
    }
    acc += block_raw * scale;
  }
  atomicAdd(out_acc + static_cast<int64_t>(tok) * kHidden + h, route_w * acc);
}

// Persistent per-process workspace: all transient device buffers grow on
// demand and are reused across calls. The per-call cudaMalloc/cudaFree cost
// was measurable on small-T workloads (observed ~61ms regression on T=1 with
// grouped path on B200); this eliminates it. Buffers leak at process exit —
// acceptable for a per-process benchmark harness.
struct MoeWorkspace {
  int64_t cap_t = 0;
  int64_t cap_t_grouped = 0;
  int64_t cap_t_small = 0;
  float* local_weight_dev = nullptr;
  uint8_t* expert_used_dev = nullptr;
  float* a_dev = nullptr;
  float* out_acc_dev = nullptr;
  int* expert_counts_dev = nullptr;
  int* expert_offsets_dev = nullptr;
  int* running_counter_dev = nullptr;
  int* permuted_token_ids_dev = nullptr;
  float* permuted_weights_dev = nullptr;
  int* permuted_experts_dev = nullptr;
  float* small_c_all_dev = nullptr;

  void ensure_core(int64_t t) {
    if (t <= cap_t && local_weight_dev != nullptr) return;
    if (local_weight_dev) cudaFree(local_weight_dev);
    if (expert_used_dev) cudaFree(expert_used_dev);
    if (a_dev) cudaFree(a_dev);
    if (out_acc_dev) cudaFree(out_acc_dev);
    cap_t = t;
    cudaMalloc(&local_weight_dev, static_cast<size_t>(t) * kNumLocalExperts * sizeof(float));
    cudaMalloc(&expert_used_dev, kNumLocalExperts * sizeof(uint8_t));
    cudaMalloc(&a_dev, static_cast<size_t>(t) * kHidden * sizeof(float));
    cudaMalloc(&out_acc_dev, static_cast<size_t>(t) * kHidden * sizeof(float));
  }

  void ensure_grouped(int64_t t) {
    if (t <= cap_t_grouped && permuted_token_ids_dev != nullptr) return;
    if (expert_counts_dev) cudaFree(expert_counts_dev);
    if (expert_offsets_dev) cudaFree(expert_offsets_dev);
    if (running_counter_dev) cudaFree(running_counter_dev);
    if (permuted_token_ids_dev) cudaFree(permuted_token_ids_dev);
    if (permuted_weights_dev) cudaFree(permuted_weights_dev);
    if (permuted_experts_dev) cudaFree(permuted_experts_dev);
    cap_t_grouped = t;
    const int64_t max_routed = t * kTopK;
    cudaMalloc(&expert_counts_dev, kNumLocalExperts * sizeof(int));
    cudaMalloc(&expert_offsets_dev, (kNumLocalExperts + 1) * sizeof(int));
    cudaMalloc(&running_counter_dev, kNumLocalExperts * sizeof(int));
    cudaMalloc(&permuted_token_ids_dev, static_cast<size_t>(max_routed) * sizeof(int));
    cudaMalloc(&permuted_weights_dev, static_cast<size_t>(max_routed) * sizeof(float));
    cudaMalloc(&permuted_experts_dev, static_cast<size_t>(max_routed) * sizeof(int));
  }

  void ensure_small(int64_t t) {
    if (t <= cap_t_small && small_c_all_dev != nullptr) return;
    if (small_c_all_dev) cudaFree(small_c_all_dev);
    cap_t_small = t;
    const int64_t max_routed = t * kTopK;
    cudaMalloc(&small_c_all_dev,
               static_cast<size_t>(max_routed) * kIntermediate * sizeof(float));
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

  // Grouped / permuted expert path: builds routing metadata and calls the
  // compact GEMM kernels instead of the dense-masked per-T loop. As of the
  // persistent-workspace + all-19 B200 deep sweep (2026-04-18) this is the
  // better default — grouped beats legacy on mean speedup (0.576x vs 0.554x
  // warmup=2 iter=10 trial=2) and wins on 13/19 workloads with losses ≤5%.
  // Opt out with FIB_MOE_LEGACY=1 for the legacy dense-mask path.
  // FIB_MOE_GROUPED=1 is still honored for backward compatibility (no-op since
  // it's already the default).
  const bool kLegacy = std::getenv("FIB_MOE_LEGACY") != nullptr;
  const bool kGrouped = !kLegacy;
  static mxfp::DeviceMxfpGemmModule gemm_mod(kHidden, kIntermediate, kBlock);
  const bool kTc = kGrouped && gemm_mod.SupportsTcPath();

  const auto t0_total =
      kProfile ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

  static MoeWorkspace ws;
  ws.ensure_core(t);
  float* local_weight_dev = ws.local_weight_dev;
  uint8_t* expert_used_dev = ws.expert_used_dev;
  float* a_dev = ws.a_dev;
  float* out_acc_dev = ws.out_acc_dev;

  cudaMemsetAsync(local_weight_dev, 0, static_cast<size_t>(t) * kNumLocalExperts * sizeof(float), stream);
  cudaMemsetAsync(expert_used_dev, 0, static_cast<size_t>(kNumLocalExperts), stream);
  cudaMemsetAsync(out_acc_dev, 0, static_cast<size_t>(t) * kHidden * sizeof(float), stream);

  int* expert_counts_dev = nullptr;
  int* expert_offsets_dev = nullptr;
  int* running_counter_dev = nullptr;
  int* permuted_token_ids_dev = nullptr;
  float* permuted_weights_dev = nullptr;
  int* permuted_experts_dev = nullptr;
  if (kGrouped) {
    ws.ensure_grouped(t);
    expert_counts_dev = ws.expert_counts_dev;
    expert_offsets_dev = ws.expert_offsets_dev;
    running_counter_dev = ws.running_counter_dev;
    permuted_token_ids_dev = ws.permuted_token_ids_dev;
    permuted_weights_dev = ws.permuted_weights_dev;
    permuted_experts_dev = ws.permuted_experts_dev;
    cudaMemsetAsync(expert_counts_dev, 0, kNumLocalExperts * sizeof(int), stream);
    cudaMemsetAsync(running_counter_dev, 0, kNumLocalExperts * sizeof(int), stream);
  }

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
      static_cast<float>(routed_scaling_factor), local_weight_dev, expert_used_dev);
  if (kProfile) {
    cudaStreamSynchronize(stream);
    t1_routing = std::chrono::high_resolution_clock::now();
  }

  int64_t n_hidden = t * kHidden;
  if (kProfile) t0_dequant = std::chrono::high_resolution_clock::now();
  if (!kTc) {
    constexpr int kDequantThreads = 256;
    int dequant_blocks = static_cast<int>((n_hidden + kDequantThreads - 1) / kDequantThreads);
    dequant_hidden_kernel<<<dequant_blocks, kDequantThreads, 0, stream>>>(
        static_cast<const uint8_t*>(hidden_states.data_ptr()),
        static_cast<const float*>(hidden_states_scale.data_ptr()), t, a_dev);
  }
  if (kProfile) {
    cudaStreamSynchronize(stream);
    t1_dequant = std::chrono::high_resolution_clock::now();
  }

  if (kProfile) t0_expert = std::chrono::high_resolution_clock::now();

  // Small-T workloads are launch-count dominated. A single all-expert Step1
  // kernel plus one scatter Step2 kernel avoids the 32-way host expert loop and
  // wins through T=80 on B200; set FIB_MOE_ALL_EXPERT_SMALL_MAX_T=0 to disable.
  int all_expert_small_max_t = 80;
  if (const char* env = std::getenv("FIB_MOE_ALL_EXPERT_SMALL_MAX_T")) {
    all_expert_small_max_t = std::atoi(env);
  }
  const bool kAllExpertSmall =
      kGrouped && !kTc && all_expert_small_max_t > 0 && t <= all_expert_small_max_t;

  if (!kTc && !kAllExpertSmall) {
    gemm_mod.EnsureWorkspace(t, stream);
  }

  std::vector<uint8_t> expert_used_host(kNumLocalExperts, 0);
  std::vector<int> expert_counts_host(kNumLocalExperts, 0);
  std::vector<int> expert_offsets_host(kNumLocalExperts + 1, 0);

  if (kGrouped) {
    // Build grouped-GEMM metadata.
    constexpr int kMetaThreads = 128;
    int meta_blocks = static_cast<int>((t + kMetaThreads - 1) / kMetaThreads);
    build_counts_kernel<<<meta_blocks, kMetaThreads, 0, stream>>>(local_weight_dev, t,
                                                                  expert_counts_dev);
    scan_offsets_kernel<<<1, 32, 0, stream>>>(expert_counts_dev, expert_offsets_dev);
    scatter_placements_kernel<<<meta_blocks, kMetaThreads, 0, stream>>>(
        local_weight_dev, t, expert_offsets_dev, running_counter_dev, permuted_token_ids_dev,
        permuted_weights_dev, permuted_experts_dev);

    if (kAllExpertSmall) {
      initialize_fp8_decode_table_once();
      ws.ensure_small(t);
      int all_expert_threads = 32;
      if (const char* env = std::getenv("FIB_MOE_ALL_EXPERT_THREADS")) {
        int parsed = std::atoi(env);
        if (parsed == 32 || parsed == 64 || parsed == 128 || parsed == 256 || parsed == 512) {
          all_expert_threads = parsed;
        }
      }
      const int64_t max_routed = t * kTopK;
      const int64_t step1_total = max_routed * kIntermediate;
      const int64_t step2_total = max_routed * kHidden;
      int step1_blocks = static_cast<int>((step1_total + all_expert_threads - 1) /
                                          all_expert_threads);
      int step2_blocks = static_cast<int>((step2_total + all_expert_threads - 1) /
                                          all_expert_threads);
      all_experts_step1_swiglu_kernel<<<step1_blocks, all_expert_threads, 0, stream>>>(
          a_dev, expert_offsets_dev, permuted_token_ids_dev, permuted_experts_dev,
          static_cast<const uint8_t*>(gemm1_weights.data_ptr()),
          static_cast<const float*>(gemm1_weights_scale.data_ptr()), max_routed,
          ws.small_c_all_dev);
      all_experts_step2_scatter_kernel<<<step2_blocks, all_expert_threads, 0, stream>>>(
          ws.small_c_all_dev, expert_offsets_dev, permuted_token_ids_dev, permuted_weights_dev,
          permuted_experts_dev, static_cast<const uint8_t*>(gemm2_weights.data_ptr()),
          static_cast<const float*>(gemm2_weights_scale.data_ptr()), max_routed, out_acc_dev);
    } else {
      cudaMemcpyAsync(expert_counts_host.data(), expert_counts_dev,
                      kNumLocalExperts * sizeof(int), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(expert_offsets_host.data(), expert_offsets_dev,
                      (kNumLocalExperts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);

      for (int le = 0; le < kNumLocalExperts; ++le) {
        int n_rows = expert_counts_host[le];
        if (n_rows == 0) continue;
        int start = expert_offsets_host[le];
        if (kTc) {
          gemm_mod.RunExpertPermutedTc(
              static_cast<const uint8_t*>(hidden_states.data_ptr()),
              static_cast<const float*>(hidden_states_scale.data_ptr()), t, n_rows,
              permuted_token_ids_dev + start, permuted_weights_dev + start, le,
              static_cast<const uint8_t*>(gemm1_weights.data_ptr()),
              static_cast<const float*>(gemm1_weights_scale.data_ptr()),
              static_cast<const uint8_t*>(gemm2_weights.data_ptr()),
              static_cast<const float*>(gemm2_weights_scale.data_ptr()), out_acc_dev, stream);
        } else {
          gemm_mod.RunExpertPermuted(
              a_dev, t, n_rows, permuted_token_ids_dev + start, permuted_weights_dev + start, le,
              static_cast<const uint8_t*>(gemm1_weights.data_ptr()),
              static_cast<const float*>(gemm1_weights_scale.data_ptr()),
              static_cast<const uint8_t*>(gemm2_weights.data_ptr()),
              static_cast<const float*>(gemm2_weights_scale.data_ptr()), out_acc_dev, stream);
        }
      }
    }
  } else {
    // Legacy dense-mask path.
    // expert_used_host drives the host-side expert loop below, so this sync is
    // functionally required and stays unconditional.
    cudaMemcpyAsync(expert_used_host.data(), expert_used_dev,
                    static_cast<size_t>(kNumLocalExperts), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int le = 0; le < kNumLocalExperts; ++le) {
      if (expert_used_host[le] == 0) continue;
      gemm_mod.RunExpert(a_dev, t, local_weight_dev, le,
                         static_cast<const uint8_t*>(gemm1_weights.data_ptr()),
                         static_cast<const float*>(gemm1_weights_scale.data_ptr()),
                         static_cast<const uint8_t*>(gemm2_weights.data_ptr()),
                         static_cast<const float*>(gemm2_weights_scale.data_ptr()), out_acc_dev,
                         stream);
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
  // Workspace is now persistent — no cudaFree here, so the previously-required
  // pre-free sync is only needed when profile timing is on.
  if (kProfile) {
    cudaStreamSynchronize(stream);
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
