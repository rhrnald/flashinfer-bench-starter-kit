#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "mxfp_gemm_module.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

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

inline bool env_enabled(const char* name) {
  const char* env = std::getenv(name);
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline int env_int(const char* name, int fallback) {
  const char* env = std::getenv(name);
  return (env == nullptr || env[0] == '\0') ? fallback : std::atoi(env);
}

struct ScopedNvtxRange {
  bool enabled;
  explicit ScopedNvtxRange(bool enabled, const char* name) : enabled(enabled) {
    if (enabled) nvtxRangePushA(name);
  }
  ~ScopedNvtxRange() {
    if (enabled) nvtxRangePop();
  }
};

struct ScopedCudaProfilerRange {
  bool active;
  ScopedCudaProfilerRange() : active(false) {
    if (!env_enabled("FIB_MOE_PROFILE_RANGE")) return;
    static std::atomic<int> call_counter{0};
    int call_idx = call_counter.fetch_add(1, std::memory_order_relaxed);
    int skip = std::max(0, env_int("FIB_MOE_PROFILE_RANGE_SKIP", 0));
    int calls = std::max(1, env_int("FIB_MOE_PROFILE_RANGE_CALLS", 1));
    active = call_idx >= skip && call_idx < skip + calls;
    if (active) cudaProfilerStart();
  }
  ~ScopedCudaProfilerRange() {
    if (active) cudaProfilerStop();
  }
};

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

__device__ __forceinline__ uint16_t float_to_bf16_rne_device(float x) {
  uint32_t u32 = __float_as_uint(x);
  uint32_t lsb = (u32 >> 16) & 1u;
  uint32_t rounding_bias = 0x7fffu + lsb;
  return static_cast<uint16_t>((u32 + rounding_bias) >> 16);
}

__device__ __forceinline__ float sigmoidf_device(float x) {
  if (x >= 0.0f) {
    float z = __expf(-x);
    return 1.0f / (1.0f + z);
  }
  float z = __expf(x);
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

template <bool GroupedMode>
__global__ void routing_kernel(const float* __restrict__ logits,
                               const uint16_t* __restrict__ bias_bf16, int64_t t,
                               int local_expert_offset, float routed_scaling_factor,
                               float* __restrict__ local_weight,
                               uint8_t* __restrict__ expert_used,
                               int* __restrict__ expert_counts,
                               int* __restrict__ routed_local_experts,
                               float* __restrict__ routed_weights,
                               int* __restrict__ routed_positions) {
  int tok = blockIdx.x * blockDim.x + threadIdx.x;
  if (tok >= t) return;

  const float* lrow = logits + static_cast<int64_t>(tok) * kNumExperts;
  float s[kNumExperts];
  for (int e = 0; e < kNumExperts; ++e) {
    s[e] = sigmoidf_device(lrow[e]);
  }

  float group_scores[kNumGroups];
  for (int g = 0; g < kNumGroups; ++g) {
    float t1 = -INFINITY;
    float t2 = -INFINITY;
    int base = g * kGroupSize;
    for (int j = 0; j < kGroupSize; ++j) {
      int e = base + j;
      float v = s[e] + bf16_to_float_device(bias_bf16[e]);
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

  TopKEntry topk[kTopK] = {
      {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1},
      {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1},
  };
  bool keep[kNumGroups] = {false, false, false, false, false, false, false, false};
  for (int i = 0; i < kTopKGroups; ++i) {
    if (grp[i].idx >= 0) keep[grp[i].idx] = true;
  }
  for (int e = 0; e < kNumExperts; ++e) {
    int g = e / kGroupSize;
    if (!keep[g]) continue;
    float v = s[e] + bf16_to_float_device(bias_bf16[e]);
    insert_topk8_device(topk, v, e);
  }

  float wsum = 1e-20f;
  for (int i = 0; i < kTopK; ++i) {
    if (topk[i].idx >= 0) wsum += s[topk[i].idx];
  }

  float* lw_row = nullptr;
  if constexpr (!GroupedMode) {
    lw_row = local_weight == nullptr ? nullptr
                                     : local_weight + static_cast<int64_t>(tok) * kNumLocalExperts;
  }
  int* routed_le_row = nullptr;
  float* routed_w_row = nullptr;
  int* routed_pos_row = nullptr;
  if constexpr (GroupedMode) {
    routed_le_row = routed_local_experts + static_cast<int64_t>(tok) * kTopK;
    routed_w_row = routed_weights + static_cast<int64_t>(tok) * kTopK;
    routed_pos_row = routed_positions + static_cast<int64_t>(tok) * kTopK;
    routed_le_row[0] = -1;
    routed_w_row[0] = 0.0f;
    routed_pos_row[0] = -1;
  } else {
    routed_le_row = routed_local_experts == nullptr
                        ? nullptr
                        : routed_local_experts + static_cast<int64_t>(tok) * kTopK;
    routed_w_row =
        routed_weights == nullptr ? nullptr : routed_weights + static_cast<int64_t>(tok) * kTopK;
    routed_pos_row = routed_positions == nullptr
                         ? nullptr
                         : routed_positions + static_cast<int64_t>(tok) * kTopK;
    if (routed_le_row != nullptr) {
      routed_le_row[0] = -1;
      routed_w_row[0] = 0.0f;
      if (routed_pos_row != nullptr) routed_pos_row[0] = -1;
    }
  }
  int local_slot = 0;
  for (int i = 0; i < kTopK; ++i) {
    int ge = topk[i].idx;
    if (ge < 0) continue;
    int le = ge - local_expert_offset;
    if (0 <= le && le < kNumLocalExperts) {
      float w = (s[ge] / wsum) * routed_scaling_factor;
      if constexpr (!GroupedMode) {
        if (lw_row != nullptr) lw_row[le] = w;
      }
      if (routed_le_row != nullptr) {
        routed_le_row[local_slot] = le;
        routed_w_row[local_slot] = w;
        if constexpr (GroupedMode) {
          routed_pos_row[local_slot] = -1;
        } else {
          if (routed_pos_row != nullptr) routed_pos_row[local_slot] = -1;
        }
      }
      ++local_slot;
      if (routed_le_row != nullptr && local_slot < kTopK) {
        routed_le_row[local_slot] = -1;
        routed_w_row[local_slot] = 0.0f;
        if constexpr (GroupedMode) {
          routed_pos_row[local_slot] = -1;
        } else {
          if (routed_pos_row != nullptr) routed_pos_row[local_slot] = -1;
        }
      }
      if constexpr (!GroupedMode) {
        if (expert_used != nullptr) expert_used[le] = 1;
        if (expert_counts != nullptr) atomicAdd(&expert_counts[le], 1);
      } else {
        atomicAdd(&expert_counts[le], 1);
      }
    }
  }
}

template <bool GroupedMode>
__global__ void routing_recompute_kernel(
    const float* __restrict__ logits, const uint16_t* __restrict__ bias_bf16, int64_t t,
    int local_expert_offset, float routed_scaling_factor, float* __restrict__ local_weight,
    uint8_t* __restrict__ expert_used, int* __restrict__ expert_counts,
    int* __restrict__ routed_local_experts, float* __restrict__ routed_weights,
    int* __restrict__ routed_positions) {
  int tok = blockIdx.x * blockDim.x + threadIdx.x;
  if (tok >= t) return;

  const float* lrow = logits + static_cast<int64_t>(tok) * kNumExperts;
  float group_scores[kNumGroups];
  for (int g = 0; g < kNumGroups; ++g) {
    float t1 = -INFINITY;
    float t2 = -INFINITY;
    int base = g * kGroupSize;
    for (int j = 0; j < kGroupSize; ++j) {
      int e = base + j;
      float v = sigmoidf_device(lrow[e]) + bf16_to_float_device(bias_bf16[e]);
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

  TopKEntry topk[kTopK] = {
      {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1},
      {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1},
  };
  bool keep[kNumGroups] = {false, false, false, false, false, false, false, false};
  for (int i = 0; i < kTopKGroups; ++i) {
    if (grp[i].idx >= 0) keep[grp[i].idx] = true;
  }
  for (int g = 0; g < kNumGroups; ++g) {
    if (!keep[g]) continue;
    int base = g * kGroupSize;
    for (int j = 0; j < kGroupSize; ++j) {
      int e = base + j;
      float v = sigmoidf_device(lrow[e]) + bf16_to_float_device(bias_bf16[e]);
      insert_topk8_device(topk, v, e);
    }
  }

  float wsum = 1e-20f;
  for (int i = 0; i < kTopK; ++i) {
    if (topk[i].idx >= 0) wsum += sigmoidf_device(lrow[topk[i].idx]);
  }

  float* lw_row = nullptr;
  if constexpr (!GroupedMode) {
    lw_row = local_weight == nullptr ? nullptr
                                     : local_weight + static_cast<int64_t>(tok) * kNumLocalExperts;
  }
  int* routed_le_row = nullptr;
  float* routed_w_row = nullptr;
  int* routed_pos_row = nullptr;
  if constexpr (GroupedMode) {
    routed_le_row = routed_local_experts + static_cast<int64_t>(tok) * kTopK;
    routed_w_row = routed_weights + static_cast<int64_t>(tok) * kTopK;
    routed_pos_row = routed_positions + static_cast<int64_t>(tok) * kTopK;
    routed_le_row[0] = -1;
    routed_w_row[0] = 0.0f;
    routed_pos_row[0] = -1;
  } else {
    routed_le_row = routed_local_experts == nullptr
                        ? nullptr
                        : routed_local_experts + static_cast<int64_t>(tok) * kTopK;
    routed_w_row =
        routed_weights == nullptr ? nullptr : routed_weights + static_cast<int64_t>(tok) * kTopK;
    routed_pos_row = routed_positions == nullptr
                         ? nullptr
                         : routed_positions + static_cast<int64_t>(tok) * kTopK;
    if (routed_le_row != nullptr) {
      routed_le_row[0] = -1;
      routed_w_row[0] = 0.0f;
      if (routed_pos_row != nullptr) routed_pos_row[0] = -1;
    }
  }
  int local_slot = 0;
  for (int i = 0; i < kTopK; ++i) {
    int ge = topk[i].idx;
    if (ge < 0) continue;
    int le = ge - local_expert_offset;
    if (0 <= le && le < kNumLocalExperts) {
      float w = (sigmoidf_device(lrow[ge]) / wsum) * routed_scaling_factor;
      if constexpr (!GroupedMode) {
        if (lw_row != nullptr) lw_row[le] = w;
      }
      if (routed_le_row != nullptr) {
        routed_le_row[local_slot] = le;
        routed_w_row[local_slot] = w;
        if constexpr (GroupedMode) {
          routed_pos_row[local_slot] = -1;
        } else {
          if (routed_pos_row != nullptr) routed_pos_row[local_slot] = -1;
        }
      }
      ++local_slot;
      if (routed_le_row != nullptr && local_slot < kTopK) {
        routed_le_row[local_slot] = -1;
        routed_w_row[local_slot] = 0.0f;
        if constexpr (GroupedMode) {
          routed_pos_row[local_slot] = -1;
        } else {
          if (routed_pos_row != nullptr) routed_pos_row[local_slot] = -1;
        }
      }
      if constexpr (!GroupedMode) {
        if (expert_used != nullptr) expert_used[le] = 1;
        if (expert_counts != nullptr) atomicAdd(&expert_counts[le], 1);
      } else {
        atomicAdd(&expert_counts[le], 1);
      }
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
//   permuted_expert_ids[num_routed]            — local expert id per slot
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
__global__ void scan_offsets_kernel(const int* __restrict__ counts, int* __restrict__ offsets,
                                    int* __restrict__ running_counter) {
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
  if (running_counter != nullptr) running_counter[tid] = 0;
  if (tid == 31) offsets[32] = incl;
}

__global__ void scatter_topk_placements_kernel(const int* __restrict__ routed_local_experts,
                                               const float* __restrict__ routed_weights,
                                               int64_t t,
                                               const int* __restrict__ expert_offsets,
                                               int* __restrict__ running_counter,
                                               int* __restrict__ permuted_token_ids,
                                               int* __restrict__ permuted_expert_ids,
                                               float* __restrict__ permuted_weights,
                                               int* __restrict__ routed_positions) {
  int tok = blockIdx.x * blockDim.x + threadIdx.x;
  if (tok >= t) return;
  const int* le_row = routed_local_experts + static_cast<int64_t>(tok) * kTopK;
  const float* w_row = routed_weights + static_cast<int64_t>(tok) * kTopK;
  #pragma unroll
  for (int i = 0; i < kTopK; ++i) {
    int le = le_row[i];
    if (le < 0) break;
    float w = w_row[i];
    int slot = atomicAdd(&running_counter[le], 1);
    int pos = expert_offsets[le] + slot;
    permuted_token_ids[pos] = tok;
    if (permuted_expert_ids != nullptr) permuted_expert_ids[pos] = le;
    if (permuted_weights != nullptr) permuted_weights[pos] = w;
    if (routed_positions != nullptr) {
      routed_positions[static_cast<int64_t>(tok) * kTopK + i] = pos;
    }
  }
}

__global__ void scatter_topk_placements_inplace_offsets_kernel(
    const int* __restrict__ routed_local_experts, const float* __restrict__ routed_weights,
    int64_t t, int* __restrict__ expert_offsets_counter,
    int* __restrict__ permuted_token_ids, int* __restrict__ permuted_expert_ids,
    float* __restrict__ permuted_weights, int* __restrict__ routed_positions) {
  int tok = blockIdx.x * blockDim.x + threadIdx.x;
  if (tok >= t) return;
  const int* le_row = routed_local_experts + static_cast<int64_t>(tok) * kTopK;
  const float* w_row = routed_weights + static_cast<int64_t>(tok) * kTopK;
#pragma unroll
  for (int i = 0; i < kTopK; ++i) {
    int le = le_row[i];
    if (le < 0) break;
    int pos = atomicAdd(&expert_offsets_counter[le], 1);
    permuted_token_ids[pos] = tok;
    if (permuted_expert_ids != nullptr) permuted_expert_ids[pos] = le;
    if (permuted_weights != nullptr) permuted_weights[pos] = w_row[i];
    if (routed_positions != nullptr) {
      routed_positions[static_cast<int64_t>(tok) * kTopK + i] = pos;
    }
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
  float* local_weight_dev = nullptr;
  uint8_t* expert_used_dev = nullptr;
  float* a_dev = nullptr;
  float* out_acc_dev = nullptr;
  int* expert_counts_dev = nullptr;
  int* expert_offsets_dev = nullptr;
  int* running_counter_dev = nullptr;
  int* routed_local_experts_dev = nullptr;
  float* routed_weights_topk_dev = nullptr;
  int* routed_positions_dev = nullptr;
  int* permuted_token_ids_dev = nullptr;
  int* permuted_expert_ids_dev = nullptr;
  float* permuted_weights_dev = nullptr;
  int* expert_offsets_host_pinned = nullptr;

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
    if (routed_local_experts_dev) cudaFree(routed_local_experts_dev);
    if (routed_weights_topk_dev) cudaFree(routed_weights_topk_dev);
    if (routed_positions_dev) cudaFree(routed_positions_dev);
    if (permuted_token_ids_dev) cudaFree(permuted_token_ids_dev);
    if (permuted_expert_ids_dev) cudaFree(permuted_expert_ids_dev);
    if (permuted_weights_dev) cudaFree(permuted_weights_dev);
    cap_t_grouped = t;
    const int64_t max_routed = t * kTopK;
    cudaMalloc(&expert_counts_dev, kNumLocalExperts * sizeof(int));
    cudaMalloc(&expert_offsets_dev, (kNumLocalExperts + 1) * sizeof(int));
    cudaMalloc(&running_counter_dev, kNumLocalExperts * sizeof(int));
    cudaMalloc(&routed_local_experts_dev, static_cast<size_t>(max_routed) * sizeof(int));
    cudaMalloc(&routed_weights_topk_dev, static_cast<size_t>(max_routed) * sizeof(float));
    cudaMalloc(&routed_positions_dev, static_cast<size_t>(max_routed) * sizeof(int));
    cudaMalloc(&permuted_token_ids_dev, static_cast<size_t>(max_routed) * sizeof(int));
    cudaMalloc(&permuted_expert_ids_dev, static_cast<size_t>(max_routed) * sizeof(int));
    cudaMalloc(&permuted_weights_dev, static_cast<size_t>(max_routed) * sizeof(float));
    if (expert_offsets_host_pinned == nullptr) {
      cudaHostAlloc(&expert_offsets_host_pinned,
                    (kNumLocalExperts + 1) * sizeof(int), cudaHostAllocDefault);
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

  const bool kNvtx = env_enabled("FIB_MOE_NVTX");
  ScopedCudaProfilerRange cuda_profiler_range;
  ScopedNvtxRange nvtx_total(kNvtx, "fib_moe_total");

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
  const char* env_direct_output = std::getenv("FIB_MOE_TC_DIRECT_OUTPUT");
  const char* env_fuse_scatter_global = std::getenv("FIB_MOE_TC_FUSE_SCATTER");
  const char* env_grouped_g1_global = std::getenv("FIB_MOE_TC_ENABLE_GROUPED_G1");
  const char* env_grouped_g1_disable = std::getenv("FIB_MOE_TC_DISABLE_GROUPED_G1");
  const char* env_grouped_g2_global = std::getenv("FIB_MOE_TC_ENABLE_GROUPED_G2");
  const bool kGroupedG1GloballyDisabled =
      env_grouped_g1_disable != nullptr && env_grouped_g1_disable[0] == '1';
  const bool kGroupedG1GloballyForced =
      !kGroupedG1GloballyDisabled && env_grouped_g1_global != nullptr &&
      env_grouped_g1_global[0] == '1';
  const bool kDirectTcOutputCandidate =
      kTc && (env_direct_output == nullptr || env_direct_output[0] != '0') &&
      (env_fuse_scatter_global == nullptr || env_fuse_scatter_global[0] != '0') &&
      (env_grouped_g2_global == nullptr || env_grouped_g2_global[0] != '1');
  bool direct_tc_output = kDirectTcOutputCandidate;
  const int dense_all_experts_max_t =
      std::max(0, env_int("FIB_MOE_TC_DENSE_ALL_EXPERTS_MAX_T", 0));
  const bool dense_all_experts_tc =
      kTc && kDirectTcOutputCandidate && dense_all_experts_max_t > 0 &&
      t <= dense_all_experts_max_t;

  const auto t0_total =
      kProfile ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

  static MoeWorkspace ws;
  ws.ensure_core(t);
  float* local_weight_dev = ws.local_weight_dev;
  uint8_t* expert_used_dev = ws.expert_used_dev;
  float* a_dev = ws.a_dev;
  float* out_acc_dev = ws.out_acc_dev;

  if (!kGrouped) {
    cudaMemsetAsync(local_weight_dev, 0,
                    static_cast<size_t>(t) * kNumLocalExperts * sizeof(float), stream);
  }
  if (!kGrouped) {
    cudaMemsetAsync(expert_used_dev, 0, static_cast<size_t>(kNumLocalExperts), stream);
  }
  if (!kGrouped || !kDirectTcOutputCandidate) {
    cudaMemsetAsync(out_acc_dev, 0, static_cast<size_t>(t) * kHidden * sizeof(float), stream);
  }

  int* expert_counts_dev = nullptr;
  int* expert_offsets_dev = nullptr;
  int* running_counter_dev = nullptr;
  int* routed_local_experts_dev = nullptr;
  float* routed_weights_topk_dev = nullptr;
  int* routed_positions_dev = nullptr;
  int* permuted_token_ids_dev = nullptr;
  int* permuted_expert_ids_dev = nullptr;
  float* permuted_weights_dev = nullptr;
  if (kGrouped) {
    ws.ensure_grouped(t);
    expert_counts_dev = ws.expert_counts_dev;
    expert_offsets_dev = ws.expert_offsets_dev;
    running_counter_dev = ws.running_counter_dev;
    routed_local_experts_dev = ws.routed_local_experts_dev;
    routed_weights_topk_dev = ws.routed_weights_topk_dev;
    routed_positions_dev = ws.routed_positions_dev;
    permuted_token_ids_dev = ws.permuted_token_ids_dev;
    permuted_expert_ids_dev = ws.permuted_expert_ids_dev;
    permuted_weights_dev = ws.permuted_weights_dev;
    cudaMemsetAsync(expert_counts_dev, 0, kNumLocalExperts * sizeof(int), stream);
  }

  std::chrono::high_resolution_clock::time_point t0_routing, t1_routing;
  std::chrono::high_resolution_clock::time_point t0_dequant, t1_dequant;
  std::chrono::high_resolution_clock::time_point t0_expert, t1_expert;
  std::chrono::high_resolution_clock::time_point t0_out, t1_out;

  if (kProfile) t0_routing = std::chrono::high_resolution_clock::now();
  {
    ScopedNvtxRange nvtx(kNvtx, "fib_moe_routing");
    constexpr int kRoutingThreads = 128;
    int routing_blocks = static_cast<int>((t + kRoutingThreads - 1) / kRoutingThreads);
    const char* env_routing_recompute = std::getenv("FIB_MOE_ROUTING_RECOMPUTE");
    const int routing_recompute_min_t =
        std::max(0, env_int("FIB_MOE_ROUTING_RECOMPUTE_MIN_T", 128));
    const bool routing_recompute_disabled =
        env_routing_recompute != nullptr && env_routing_recompute[0] == '0';
    const bool routing_recompute_enabled =
        !routing_recompute_disabled &&
        ((env_routing_recompute != nullptr && env_routing_recompute[0] == '1') ||
         (routing_recompute_min_t > 0 && t >= routing_recompute_min_t));
    if (routing_recompute_enabled) {
      if (kGrouped) {
        routing_recompute_kernel<true><<<routing_blocks, kRoutingThreads, 0, stream>>>(
            static_cast<const float*>(routing_logits.data_ptr()),
            static_cast<const uint16_t*>(routing_bias.data_ptr()), t,
            static_cast<int>(local_expert_offset), static_cast<float>(routed_scaling_factor),
            nullptr, nullptr, expert_counts_dev, routed_local_experts_dev, routed_weights_topk_dev,
            routed_positions_dev);
      } else {
        routing_recompute_kernel<false><<<routing_blocks, kRoutingThreads, 0, stream>>>(
            static_cast<const float*>(routing_logits.data_ptr()),
            static_cast<const uint16_t*>(routing_bias.data_ptr()), t,
            static_cast<int>(local_expert_offset), static_cast<float>(routed_scaling_factor),
            local_weight_dev, expert_used_dev, nullptr, routed_local_experts_dev,
            routed_weights_topk_dev, routed_positions_dev);
      }
    } else {
      if (kGrouped) {
        routing_kernel<true><<<routing_blocks, kRoutingThreads, 0, stream>>>(
            static_cast<const float*>(routing_logits.data_ptr()),
            static_cast<const uint16_t*>(routing_bias.data_ptr()), t,
            static_cast<int>(local_expert_offset), static_cast<float>(routed_scaling_factor),
            nullptr, nullptr, expert_counts_dev, routed_local_experts_dev, routed_weights_topk_dev,
            routed_positions_dev);
      } else {
        routing_kernel<false><<<routing_blocks, kRoutingThreads, 0, stream>>>(
            static_cast<const float*>(routing_logits.data_ptr()),
            static_cast<const uint16_t*>(routing_bias.data_ptr()), t,
            static_cast<int>(local_expert_offset), static_cast<float>(routed_scaling_factor),
            local_weight_dev, expert_used_dev, nullptr, routed_local_experts_dev,
            routed_weights_topk_dev, routed_positions_dev);
      }
    }
  }
  if (kProfile) {
    cudaStreamSynchronize(stream);
    t1_routing = std::chrono::high_resolution_clock::now();
  }

  int64_t n_hidden = t * kHidden;
  if (kProfile) t0_dequant = std::chrono::high_resolution_clock::now();
  if (!kTc) {
    ScopedNvtxRange nvtx(kNvtx, "fib_moe_dequant_hidden");
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

  if (!kTc) {
    gemm_mod.EnsureWorkspace(t, stream);
  }

  std::array<uint8_t, kNumLocalExperts> expert_used_host{};
  std::array<int, kNumLocalExperts> expert_counts_host{};
  std::array<int, kNumLocalExperts + 1> expert_offsets_host_fallback{};
  int* expert_offsets_host =
      (kGrouped && ws.expert_offsets_host_pinned != nullptr) ? ws.expert_offsets_host_pinned
                                                             : expert_offsets_host_fallback.data();

  if (dense_all_experts_tc) {
    ScopedNvtxRange nvtx(kNvtx, "fib_moe_dense_all_experts_tc");
    bool ok = gemm_mod.RunAllExpertsDenseTc(
        static_cast<const uint8_t*>(hidden_states.data_ptr()),
        static_cast<const float*>(hidden_states_scale.data_ptr()), t, routed_local_experts_dev,
        routed_weights_topk_dev, static_cast<const uint8_t*>(gemm1_weights.data_ptr()),
        static_cast<const float*>(gemm1_weights_scale.data_ptr()),
        static_cast<const uint8_t*>(gemm2_weights.data_ptr()),
        static_cast<const float*>(gemm2_weights_scale.data_ptr()),
        static_cast<uint16_t*>(output.data_ptr()), stream);
    if (ok) return;
  }

  if (kGrouped) {
    ScopedNvtxRange nvtx_meta(kNvtx, "fib_moe_metadata_and_experts");
    // Build grouped-GEMM metadata.
    constexpr int kMetaThreads = 128;
    int meta_blocks = static_cast<int>((t + kMetaThreads - 1) / kMetaThreads);
    {
      ScopedNvtxRange nvtx(kNvtx, "fib_moe_metadata");
      scan_offsets_kernel<<<1, 32, 0, stream>>>(expert_counts_dev, expert_offsets_dev,
                                                running_counter_dev);
      cudaMemcpyAsync(expert_offsets_host, expert_offsets_dev,
                      (kNumLocalExperts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      for (int le = 0; le < kNumLocalExperts; ++le) {
        expert_counts_host[le] = expert_offsets_host[le + 1] - expert_offsets_host[le];
      }
      const int total_routed_rows = expert_offsets_host[kNumLocalExperts];
      const bool grouped_g1_forced = kGroupedG1GloballyForced;
      const bool grouped_g2_forced =
          env_grouped_g2_global != nullptr && env_grouped_g2_global[0] == '1';
      const bool grouped_g1_auto_allowed =
          kTc && env_grouped_g1_global == nullptr && !kGroupedG1GloballyDisabled;
      const int grouped_g1_auto_max_rows =
          std::max(0, env_int("FIB_MOE_TC_GROUPED_G1_AUTO_MAX_ROWS", 32768));
      const bool grouped_g1_auto_candidate =
          grouped_g1_auto_allowed && grouped_g1_auto_max_rows > 0 &&
          total_routed_rows <= grouped_g1_auto_max_rows;
      const bool grouped_gemm_candidate =
          kTc && (grouped_g1_forced || grouped_g1_auto_candidate || grouped_g2_forced);
      direct_tc_output = kDirectTcOutputCandidate && !grouped_g2_forced;
      if (!direct_tc_output && kDirectTcOutputCandidate) {
        cudaMemsetAsync(out_acc_dev, 0, static_cast<size_t>(t) * kHidden * sizeof(float), stream);
      }
      const bool preserve_device_offsets =
          kTc && (!direct_tc_output || grouped_g1_forced || grouped_g1_auto_candidate ||
                  grouped_g2_forced);
      const char* env_meta_inplace_offsets = std::getenv("FIB_MOE_META_INPLACE_OFFSETS");
      const bool use_inplace_offsets =
          env_meta_inplace_offsets == nullptr || env_meta_inplace_offsets[0] != '0';
      if (preserve_device_offsets || !use_inplace_offsets) {
        scatter_topk_placements_kernel<<<meta_blocks, kMetaThreads, 0, stream>>>(
            routed_local_experts_dev, routed_weights_topk_dev, t, expert_offsets_dev,
            running_counter_dev, permuted_token_ids_dev,
            direct_tc_output ? nullptr : permuted_expert_ids_dev,
            direct_tc_output ? nullptr : permuted_weights_dev, routed_positions_dev);
      } else {
        scatter_topk_placements_inplace_offsets_kernel<<<meta_blocks, kMetaThreads, 0, stream>>>(
            routed_local_experts_dev, routed_weights_topk_dev, t, expert_offsets_dev,
            permuted_token_ids_dev, direct_tc_output ? nullptr : permuted_expert_ids_dev,
            direct_tc_output ? nullptr : permuted_weights_dev,
            routed_positions_dev);
      }
      (void)grouped_gemm_candidate;
    }
    if (kProfile) {
      int min_rows = std::numeric_limits<int>::max();
      int max_rows = 0;
      int lt16 = 0;
      int lt32 = 0;
      int lt64 = 0;
      for (int le = 0; le < kNumLocalExperts; ++le) {
        int n_rows = expert_counts_host[le];
        min_rows = std::min(min_rows, n_rows);
        max_rows = std::max(max_rows, n_rows);
        lt16 += (n_rows < 16);
        lt32 += (n_rows < 32);
        lt64 += (n_rows < 64);
      }
      std::fprintf(stderr,
                   "[moe_grouped] seq_len=%lld total_rows=%d min_rows=%d max_rows=%d lt16=%d lt32=%d lt64=%d\n",
                   static_cast<long long>(t), expert_offsets_host[kNumLocalExperts], min_rows,
                   max_rows, lt16, lt32, lt64);
    }

    bool grouped_g1_done = false;
    bool grouped_g2_done = false;
    {
      ScopedNvtxRange nvtx(kNvtx, "fib_moe_expert_compute");
      if (kTc) {
        grouped_g1_done = gemm_mod.RunGroupedGemm1ThenExpertGemm2Tc(
            static_cast<const uint8_t*>(hidden_states.data_ptr()),
            static_cast<const float*>(hidden_states_scale.data_ptr()), t,
            expert_offsets_host[kNumLocalExperts], expert_offsets_dev, expert_counts_host.data(),
            expert_offsets_host, permuted_token_ids_dev, permuted_weights_dev,
            static_cast<const uint8_t*>(gemm1_weights.data_ptr()),
            static_cast<const float*>(gemm1_weights_scale.data_ptr()),
            static_cast<const uint8_t*>(gemm2_weights.data_ptr()),
            static_cast<const float*>(gemm2_weights_scale.data_ptr()), out_acc_dev, stream,
            direct_tc_output, routed_positions_dev, routed_local_experts_dev,
            routed_weights_topk_dev, static_cast<uint16_t*>(output.data_ptr()));
      }
      if (kTc && !grouped_g1_done) {
        grouped_g2_done = gemm_mod.RunExpertGemm1ThenGroupedGemm2Tc(
            static_cast<const uint8_t*>(hidden_states.data_ptr()),
            static_cast<const float*>(hidden_states_scale.data_ptr()), t,
            expert_offsets_host[kNumLocalExperts], expert_offsets_dev, expert_counts_host.data(),
            expert_offsets_host, permuted_token_ids_dev, permuted_expert_ids_dev,
            permuted_weights_dev,
            static_cast<const uint8_t*>(gemm1_weights.data_ptr()),
            static_cast<const float*>(gemm1_weights_scale.data_ptr()),
            static_cast<const uint8_t*>(gemm2_weights.data_ptr()),
            static_cast<const float*>(gemm2_weights_scale.data_ptr()), out_acc_dev, stream);
      }

      const bool fuse_scatter =
          kTc && !grouped_g1_done && !grouped_g2_done &&
          (env_fuse_scatter_global == nullptr || env_fuse_scatter_global[0] != '0');
      for (int le = 0; le < kNumLocalExperts && !grouped_g1_done && !grouped_g2_done; ++le) {
        int n_rows = expert_counts_host[le];
        if (n_rows == 0) continue;
        int start = expert_offsets_host[le];
        if (kProfile) {
          int small_rows_threshold = 0;
          const char* env_small = std::getenv("FIB_MOE_TC_SMALL_ROWS");
          if (env_small != nullptr && env_small[0] != '\0') {
            small_rows_threshold = std::max(0, std::atoi(env_small));
          }
          const char* path = !kTc ? "cuda_compact"
                                  : (n_rows <= small_rows_threshold ? "cuda_small"
                                                                    : "cutlass_fp8_g1_f16_g2");
          std::fprintf(stderr, "[moe_expert] seq_len=%lld le=%d n_rows=%d path=%s\n",
                       static_cast<long long>(t), le, n_rows, path);
        }
        if (fuse_scatter) {
          int scratch_start = start + 4 * le;
          gemm_mod.RunExpertPermutedTcToScratch(
              static_cast<const uint8_t*>(hidden_states.data_ptr()),
              static_cast<const float*>(hidden_states_scale.data_ptr()), t,
              expert_offsets_host[kNumLocalExperts] + 4 * kNumLocalExperts, scratch_start, n_rows,
              permuted_token_ids_dev + start, le,
              static_cast<const uint8_t*>(gemm1_weights.data_ptr()),
              static_cast<const float*>(gemm1_weights_scale.data_ptr()),
              static_cast<const uint8_t*>(gemm2_weights.data_ptr()),
              static_cast<const float*>(gemm2_weights_scale.data_ptr()), stream);
        } else if (kTc) {
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
      if (fuse_scatter) {
        ScopedNvtxRange nvtx_out(kNvtx, "fib_moe_direct_output_or_scatter");
        if (direct_tc_output) {
          gemm_mod.WriteTcScratchToBf16Output(
              t, routed_positions_dev, routed_local_experts_dev, routed_weights_topk_dev,
              static_cast<uint16_t*>(output.data_ptr()), stream);
        } else {
          gemm_mod.ScatterTcScratch(expert_offsets_host[kNumLocalExperts], expert_offsets_dev,
                                    permuted_token_ids_dev, permuted_expert_ids_dev,
                                    permuted_weights_dev, out_acc_dev, stream);
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
  if (!direct_tc_output) {
    constexpr int kOutThreads = 256;
    int out_blocks = static_cast<int>((n_hidden + kOutThreads - 1) / kOutThreads);
    f32_to_bf16_kernel<<<out_blocks, kOutThreads, 0, stream>>>(
        out_acc_dev, n_hidden, static_cast<uint16_t*>(output.data_ptr()));
  }
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
