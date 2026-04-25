#include "mxfp_gemm_module.h"
#include "mxfp_cutlass_sm100.cuh"
#include "mxfp_device_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <array>
#include <vector>

namespace mxfp {

namespace {

struct TcGemmDispatch {
  int tile_m;
  int mma_sms;
};

bool mxfp_nvtx_enabled() {
  const char* env = std::getenv("FIB_MOE_NVTX");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

struct ScopedMxfpNvtxRange {
  bool enabled;
  explicit ScopedMxfpNvtxRange(bool enabled, const char* name) : enabled(enabled) {
    if (enabled) nvtxRangePushA(name);
  }
  ~ScopedMxfpNvtxRange() {
    if (enabled) nvtxRangePop();
  }
};

int next_power_of_two_int(int n) {
  int v = std::max(1, n);
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

int compute_trtllm_tile_tokens_dim(int total_rows, int num_experts = 32) {
  int avg_rows = std::max(1, (total_rows + num_experts - 1) / num_experts);
  int adjusted = (avg_rows * 13 + 9) / 10;
  int tile = next_power_of_two_int(adjusted);
  return std::min(std::max(tile, 8), 128);
}

TcGemmDispatch select_trtllm_like_gemm1_dispatch(int total_rows, int padded_rows) {
  if (padded_rows >= 256) return {256, 2};
  (void)total_rows;
  return {64, 1};
}

TcGemmDispatch select_trtllm_like_gemm2_dispatch(int total_rows, int padded_rows) {
  (void)total_rows;
  (void)padded_rows;
  return {64, 1};
}

int select_tile_m_from_env(const char* name, int fallback) {
  const char* env = std::getenv(name);
  if (env == nullptr) return fallback;
  int v = std::atoi(env);
  return (v == 64 || v == 128 || v == 256) ? v : fallback;
}

int select_mma_sm_from_env_value(const char* env, int fallback) {
  if (env == nullptr) return fallback;
  int v = std::atoi(env);
  return (v == 1 || v == 2) ? v : fallback;
}

int select_mma_sm_from_env(const char* name, int fallback) {
  return select_mma_sm_from_env_value(std::getenv(name), fallback);
}

int select_swiglu_rows_per_cta(int total_rows, int padded_rows) {
  const char* env = std::getenv("FIB_MOE_SWIGLU_ROWS_PER_CTA");
  if (env != nullptr) {
    int v = std::atoi(env);
    if (v == 1 || v == 2 || v == 4) return v;
  }
  (void)total_rows;
  (void)padded_rows;
  return 1;
}

int select_grouped_swiglu_rows_per_cta(int total_rows, int padded_rows) {
  const char* env = std::getenv("FIB_MOE_TC_GROUPED_SWIGLU_ROWS_PER_CTA");
  if (env != nullptr) {
    int v = std::atoi(env);
    if (v == 1 || v == 2 || v == 4) return v;
  }
  (void)total_rows;
  (void)padded_rows;
  return 1;
}

int select_grouped_swiglu_col_blocks_per_cta(int total_rows, int padded_rows) {
  const char* env = std::getenv("FIB_MOE_TC_GROUPED_SWIGLU_COL_BLOCKS_PER_CTA");
  if (env != nullptr) {
    int v = std::atoi(env);
    if (v == 1 || v == 2 || v == 4) return v;
  }
  (void)total_rows;
  (void)padded_rows;
  return 1;
}

int select_direct_output_vec(int64_t t) {
  const char* env = std::getenv("FIB_MOE_DIRECT_OUTPUT_VEC");
  if (env != nullptr) {
    int v = std::atoi(env);
    if (v == 1 || v == 2 || v == 4 || v == 8) return v;
  }
  const char* env_min_t = std::getenv("FIB_MOE_DIRECT_OUTPUT_VEC_MIN_T");
  int min_t = env_min_t == nullptr ? 64 : std::max(0, std::atoi(env_min_t));
  return t >= min_t ? 4 : 1;
}

int select_direct_output_threads(int64_t t) {
  const char* env = std::getenv("FIB_MOE_DIRECT_OUTPUT_THREADS");
  if (env != nullptr) {
    int v = std::atoi(env);
    if (v == 64 || v == 128 || v == 256 || v == 512) return v;
  }
  return (t >= 50 && t <= 64) ? 64 : 128;
}

bool use_vec16_hidden_gather(int rows) {
  const char* env_min_rows = std::getenv("FIB_MOE_HIDDEN_GATHER_VEC16_MIN_ROWS");
  int min_rows = env_min_rows == nullptr ? 0 : std::max(0, std::atoi(env_min_rows));
  return rows >= min_rows;
}

#if FIB_HAS_DIRECT_CUTLASS_SM100
#if FIB_HAS_FLASHINFER_GROUP_GEMM_FP8_SM100
using detail::launch_flashinfer_grouped_blockscaled_gemm1_sm100;
#endif
using detail::launch_cutlass_blockscaled_group_gemm_sm100;
using detail::launch_cutlass_blockscaled_group_gemm_tn_sm100;
using detail::launch_cutlass_blockscaled_grouped_ptr_gemm_sm100;
using detail::launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_select_epilogue;
using detail::launch_cutlass_dense_gemm_sm100;
using detail::launch_cutlass_dense_grouped_ptr_gemm_sm100;
#endif
using detail::bf16_to_float_device;
using detail::env_int_or_default;
using detail::f16_to_float_device;
using detail::float_to_e4m3_device;
using detail::fp8_e4m3fn_to_float_device;
using detail::fp8_native_to_float_device;
using detail::quantize_e4m3fn_like;
using detail::siluf_device;

__global__ void gemm1_kernel(const float* __restrict__ a, int64_t t, int hidden, int gemm1_out,
                             int block, int hidden_blocks, int local_expert_idx,
                             const float* __restrict__ local_weight, const uint8_t* __restrict__ w13,
                             const float* __restrict__ s13, bool emulate_fp8_unit,
                             bool emulate_fp16_operands, bool emulate_acc_half,
                             float* __restrict__ g1) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = t * gemm1_out;
  if (idx >= total) return;

  int64_t tok = idx / gemm1_out;
  int j = static_cast<int>(idx - tok * gemm1_out);

  float w_tok = local_weight[tok * 32 + local_expert_idx];
  if (w_tok == 0.0f) {
    g1[idx] = 0.0f;
    return;
  }

  int jb = j / block;
  const float* a_row = a + tok * hidden;
  const uint8_t* w_row = w13 + static_cast<int64_t>(j) * hidden;
  float acc = 0.0f;
  __half acc_h = __float2half(0.0f);
  for (int hb = 0; hb < hidden_blocks; ++hb) {
    float scale = s13[jb * hidden_blocks + hb];
    float block_raw = 0.0f;
    int h0 = hb * block;
    for (int u = 0; u < block; ++u) {
      int h = h0 + u;
      float wv_raw = fp8_e4m3fn_to_float_device(w_row[h]);
      float av = a_row[h];
      if (emulate_fp8_unit) {
        av = quantize_e4m3fn_like(av);
        wv_raw = quantize_e4m3fn_like(wv_raw);
      }
      float prod_raw;
      if (emulate_fp16_operands) {
        __half av_h = __float2half(av);
        __half wv_h = __float2half(wv_raw);
        prod_raw = __half2float(__hmul(av_h, wv_h));
      } else {
        prod_raw = av * wv_raw;
      }
      block_raw += prod_raw;
    }
    float block_val = block_raw * scale;
    if (emulate_acc_half) {
      // Optional narrow-accumulator mode.
      acc_h = __hadd(acc_h, __float2half(block_val));
    } else {
      acc += block_val;
    }
  }
  g1[idx] = emulate_acc_half ? __half2float(acc_h) : acc;
}

__global__ void swiglu_kernel(const float* __restrict__ g1, int64_t t, int intermediate,
                              int local_expert_idx, const float* __restrict__ local_weight,
                              bool emulate_fp8_unit, float* __restrict__ c) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = t * intermediate;
  if (idx >= total) return;
  int64_t tok = idx / intermediate;
  int i = static_cast<int>(idx - tok * intermediate);

  float w_tok = local_weight[tok * 32 + local_expert_idx];
  if (w_tok == 0.0f) {
    c[idx] = 0.0f;
    return;
  }

  const float* g1_row = g1 + tok * (2 * intermediate);
  float x1 = g1_row[i];
  float x2 = g1_row[i + intermediate];
  float y = x1 * siluf_device(x2);
  // Keep FP32 activation path in TC-like emulation mode.
  (void)emulate_fp8_unit;
  c[idx] = y;
}

// Permuted-path GEMM1. `a` is the original [T, H] activation tensor; we gather
// rows via `permuted_tok[pr]` rather than expanding into a compact buffer, to
// save a DRAM pass. Output is compact: `g1_perm[n_rows, gemm1_out]`.
__global__ void gemm1_permuted_kernel(const float* __restrict__ a, int hidden, int gemm1_out,
                                      int block, int hidden_blocks, int n_rows,
                                      const int* __restrict__ permuted_tok,
                                      const uint8_t* __restrict__ w13,
                                      const float* __restrict__ s13, bool emulate_fp8_unit,
                                      bool emulate_fp16_operands, bool emulate_acc_half,
                                      float* __restrict__ g1_perm) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * gemm1_out;
  if (idx >= total) return;

  int pr = static_cast<int>(idx / gemm1_out);
  int j = static_cast<int>(idx - static_cast<int64_t>(pr) * gemm1_out);
  int tok = permuted_tok[pr];

  int jb = j / block;
  const float* a_row = a + static_cast<int64_t>(tok) * hidden;
  const uint8_t* w_row = w13 + static_cast<int64_t>(j) * hidden;
  float acc = 0.0f;
  __half acc_h = __float2half(0.0f);
  for (int hb = 0; hb < hidden_blocks; ++hb) {
    float scale = s13[jb * hidden_blocks + hb];
    float block_raw = 0.0f;
    int h0 = hb * block;
    for (int u = 0; u < block; ++u) {
      int h = h0 + u;
      float wv_raw = fp8_e4m3fn_to_float_device(w_row[h]);
      float av = a_row[h];
      if (emulate_fp8_unit) {
        av = quantize_e4m3fn_like(av);
        wv_raw = quantize_e4m3fn_like(wv_raw);
      }
      float prod_raw;
      if (emulate_fp16_operands) {
        __half av_h = __float2half(av);
        __half wv_h = __float2half(wv_raw);
        prod_raw = __half2float(__hmul(av_h, wv_h));
      } else {
        prod_raw = av * wv_raw;
      }
      block_raw += prod_raw;
    }
    float block_val = block_raw * scale;
    if (emulate_acc_half) {
      acc_h = __hadd(acc_h, __float2half(block_val));
    } else {
      acc += block_val;
    }
  }
  g1_perm[idx] = emulate_acc_half ? __half2float(acc_h) : acc;
}

__global__ void gemm1_compact_kernel(const float* __restrict__ a_compact, int hidden,
                                     int gemm1_out, int block, int hidden_blocks, int n_rows,
                                     const uint8_t* __restrict__ w13,
                                     const float* __restrict__ s13, bool emulate_fp8_unit,
                                     bool emulate_fp16_operands, bool emulate_acc_half,
                                     float* __restrict__ g1_out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * gemm1_out;
  if (idx >= total) return;

  int row = static_cast<int>(idx / gemm1_out);
  int j = static_cast<int>(idx - static_cast<int64_t>(row) * gemm1_out);
  int jb = j / block;

  const float* a_row = a_compact + static_cast<int64_t>(row) * hidden;
  const uint8_t* w_row = w13 + static_cast<int64_t>(j) * hidden;
  float acc = 0.0f;
  __half acc_h = __float2half(0.0f);
  for (int hb = 0; hb < hidden_blocks; ++hb) {
    float scale = s13[jb * hidden_blocks + hb];
    float block_raw = 0.0f;
    int h0 = hb * block;
    for (int u = 0; u < block; ++u) {
      int h = h0 + u;
      float wv_raw = fp8_e4m3fn_to_float_device(w_row[h]);
      float av = a_row[h];
      if (emulate_fp8_unit) {
        av = quantize_e4m3fn_like(av);
        wv_raw = quantize_e4m3fn_like(wv_raw);
      }
      float prod_raw;
      if (emulate_fp16_operands) {
        __half av_h = __float2half(av);
        __half wv_h = __float2half(wv_raw);
        prod_raw = __half2float(__hmul(av_h, wv_h));
      } else {
        prod_raw = av * wv_raw;
      }
      block_raw += prod_raw;
    }
    float block_val = block_raw * scale;
    if (emulate_acc_half) {
      acc_h = __hadd(acc_h, __float2half(block_val));
    } else {
      acc += block_val;
    }
  }
  g1_out[idx] = emulate_acc_half ? __half2float(acc_h) : acc;
}

// Permuted swiglu: input/output are both compact n_rows-indexed, no masking.
__global__ void swiglu_permuted_kernel(const float* __restrict__ g1_perm, int intermediate,
                                       int n_rows, bool emulate_fp8_unit,
                                       float* __restrict__ c_perm) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * intermediate;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / intermediate);
  int i = static_cast<int>(idx - static_cast<int64_t>(pr) * intermediate);
  const float* g1_row = g1_perm + static_cast<int64_t>(pr) * (2 * intermediate);
  float x1 = g1_row[i];
  float x2 = g1_row[i + intermediate];
  (void)emulate_fp8_unit;  // Activation path stays FP32 in TC-like emulation.
  c_perm[idx] = x1 * siluf_device(x2);
}

// Permuted GEMM2 + scatter-accumulate into the global [T, H] out_acc tensor.
// Within a single launch each (pr, h) touches a unique out_acc cell because a
// token routes to any given expert at most once; across expert launches the
// stream orders contributions, so no atomicAdd is needed.
__global__ void gemm2_scatter_accumulate_kernel(const float* __restrict__ c_perm, int hidden,
                                                int intermediate, int block,
                                                int intermediate_blocks, int n_rows,
                                                const int* __restrict__ permuted_tok,
                                                const float* __restrict__ permuted_w,
                                                const uint8_t* __restrict__ w2,
                                                const float* __restrict__ s2,
                                                bool emulate_fp8_unit,
                                                bool emulate_fp16_operands, bool emulate_acc_half,
                                                float* __restrict__ out_acc) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * hidden;
  if (idx >= total) return;

  int pr = static_cast<int>(idx / hidden);
  int h = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden);
  int tok = permuted_tok[pr];
  float w_tok = permuted_w[pr];

  int hb = h / block;
  const float* c_row = c_perm + static_cast<int64_t>(pr) * intermediate;
  const uint8_t* w_row = w2 + static_cast<int64_t>(h) * intermediate;
  float acc = 0.0f;
  __half acc_h = __float2half(0.0f);
  for (int ib = 0; ib < intermediate_blocks; ++ib) {
    float scale = s2[hb * intermediate_blocks + ib];
    float block_raw = 0.0f;
    int i0 = ib * block;
    for (int u = 0; u < block; ++u) {
      int i = i0 + u;
      float wv_raw = fp8_e4m3fn_to_float_device(w_row[i]);
      float cv = c_row[i];
      if (emulate_fp8_unit) {
        cv = quantize_e4m3fn_like(cv);
        wv_raw = quantize_e4m3fn_like(wv_raw);
      }
      float prod_raw;
      if (emulate_fp16_operands) {
        __half cv_h = __float2half(cv);
        __half wv_h = __float2half(wv_raw);
        prod_raw = __half2float(__hmul(cv_h, wv_h));
      } else {
        prod_raw = cv * wv_raw;
      }
      block_raw += prod_raw;
    }
    float block_val = block_raw * scale;
    if (emulate_acc_half) {
      acc_h = __hadd(acc_h, __float2half(block_val));
    } else {
      acc += block_val;
    }
  }
  acc = emulate_acc_half ? __half2float(acc_h) : acc;
  out_acc[static_cast<int64_t>(tok) * hidden + h] += w_tok * acc;
}

__global__ void gemm2_acc_kernel(const float* __restrict__ c, int64_t t, int hidden, int intermediate,
                                 int block, int intermediate_blocks, int local_expert_idx,
                                 const float* __restrict__ local_weight, const uint8_t* __restrict__ w2,
                                 const float* __restrict__ s2, bool emulate_fp8_unit,
                                 bool emulate_fp16_operands, bool emulate_acc_half,
                                 float* __restrict__ out_acc) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = t * hidden;
  if (idx >= total) return;

  int64_t tok = idx / hidden;
  int h = static_cast<int>(idx - tok * hidden);

  float w_tok = local_weight[tok * 32 + local_expert_idx];
  if (w_tok == 0.0f) return;

  int hb = h / block;
  const float* c_row = c + tok * intermediate;
  const uint8_t* w_row = w2 + static_cast<int64_t>(h) * intermediate;
  float acc = 0.0f;
  __half acc_h = __float2half(0.0f);
  for (int ib = 0; ib < intermediate_blocks; ++ib) {
    float scale = s2[hb * intermediate_blocks + ib];
    float block_raw = 0.0f;
    int i0 = ib * block;
    for (int u = 0; u < block; ++u) {
      int i = i0 + u;
      float wv_raw = fp8_e4m3fn_to_float_device(w_row[i]);
      float cv = c_row[i];
      if (emulate_fp8_unit) {
        cv = quantize_e4m3fn_like(cv);
        wv_raw = quantize_e4m3fn_like(wv_raw);
      }
      float prod_raw;
      if (emulate_fp16_operands) {
        __half cv_h = __float2half(cv);
        __half wv_h = __float2half(wv_raw);
        prod_raw = __half2float(__hmul(cv_h, wv_h));
      } else {
        prod_raw = cv * wv_raw;
      }
      block_raw += prod_raw;
    }
    float block_val = block_raw * scale;
    if (emulate_acc_half) {
      // Optional narrow-accumulator mode.
      acc_h = __hadd(acc_h, __float2half(block_val));
    } else {
      acc += block_val;
    }
  }

  acc = emulate_acc_half ? __half2float(acc_h) : acc;
  out_acc[idx] += w_tok * acc;
}

__global__ void write_single_group_indptr_kernel(int padded_rows, int* __restrict__ indptr) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    indptr[0] = 0;
    indptr[1] = padded_rows;
  }
}

__global__ void build_padded_offsets_from_expert_offsets_kernel(
    const int* __restrict__ expert_offsets, int* __restrict__ padded_offsets) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int running = 0;
  padded_offsets[0] = 0;
#pragma unroll
  for (int le = 0; le < 32; ++le) {
    int n_rows = expert_offsets[le + 1] - expert_offsets[le];
    running += (n_rows + 3) & ~3;
    padded_offsets[le + 1] = running;
  }
}

__global__ void build_padded_compact_row_map_kernel(const int* __restrict__ expert_offsets,
                                                    const int* __restrict__ padded_offsets,
                                                    int* __restrict__ compact_rows) {
  int le = blockIdx.x;
  int n_rows = expert_offsets[le + 1] - expert_offsets[le];
  int padded_rows = padded_offsets[le + 1] - padded_offsets[le];
  int padded_base = padded_offsets[le];
  int compact_base = expert_offsets[le];
  for (int r = threadIdx.x; r < padded_rows; r += blockDim.x) {
    compact_rows[padded_base + r] = (r < n_rows) ? (compact_base + r) : -1;
  }
}

__global__ void gather_hidden_fp8_rows_kernel(const uint8_t* __restrict__ hidden_fp8,
                                              const int* __restrict__ permuted_tok,
                                              int n_rows, int padded_rows, int hidden,
                                             uint8_t* __restrict__ a_perm) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(padded_rows) * hidden;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / hidden);
  int h = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden);
  if (pr >= n_rows) {
    a_perm[idx] = 0;
    return;
  }
  int tok = permuted_tok[pr];
  a_perm[idx] = hidden_fp8[static_cast<int64_t>(tok) * hidden + h];
}

__global__ void gather_hidden_scale_rows_kernel(const float* __restrict__ hidden_scale,
                                               const int* __restrict__ permuted_tok,
                                               int64_t t, int n_rows, int padded_rows,
                                               int hidden_blocks, float* __restrict__ a_scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(padded_rows) * hidden_blocks;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / hidden_blocks);
  int hb = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden_blocks);
  if (pr >= n_rows) {
    a_scale[static_cast<int64_t>(hb) * padded_rows + pr] = 1.0f;
    return;
  }
  int tok = permuted_tok[pr];
  a_scale[static_cast<int64_t>(hb) * padded_rows + pr] =
      hidden_scale[static_cast<int64_t>(hb) * t + tok];
}

__global__ void gather_hidden_scale_rows_mn_major_kernel(const float* __restrict__ hidden_scale,
                                                         const int* __restrict__ permuted_tok,
                                                         int64_t t, int n_rows, int padded_rows,
                                                         int hidden_blocks,
                                                         float* __restrict__ a_scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(padded_rows) * hidden_blocks;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / hidden_blocks);
  int hb = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden_blocks);
  if (pr >= n_rows) {
    a_scale[static_cast<int64_t>(pr) * hidden_blocks + hb] = 1.0f;
    return;
  }
  int tok = permuted_tok[pr];
  a_scale[static_cast<int64_t>(pr) * hidden_blocks + hb] =
      hidden_scale[static_cast<int64_t>(hb) * t + tok];
}

__global__ void gather_hidden_fp8_and_scale_rows_mn_major_kernel(
    const uint8_t* __restrict__ hidden_fp8, const float* __restrict__ hidden_scale,
    const int* __restrict__ permuted_tok, int64_t t, int n_rows, int padded_rows, int hidden,
    int hidden_blocks, uint8_t* __restrict__ a_perm, float* __restrict__ a_scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t a_total = static_cast<int64_t>(padded_rows) * hidden;
  int64_t scale_total = static_cast<int64_t>(padded_rows) * hidden_blocks;

  if (idx < a_total) {
    int pr = static_cast<int>(idx / hidden);
    int h = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden);
    a_perm[idx] = (pr < n_rows) ? hidden_fp8[static_cast<int64_t>(permuted_tok[pr]) * hidden + h]
                                : 0;
  }

  if (idx < scale_total) {
    int pr = static_cast<int>(idx / hidden_blocks);
    int hb = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden_blocks);
    a_scale[static_cast<int64_t>(pr) * hidden_blocks + hb] =
        (pr < n_rows) ? hidden_scale[static_cast<int64_t>(hb) * t + permuted_tok[pr]] : 1.0f;
  }
}

__global__ void gather_hidden_fp8_vec16_and_scale_rows_mn_major_kernel(
    const uint8_t* __restrict__ hidden_fp8, const float* __restrict__ hidden_scale,
    const int* __restrict__ permuted_tok, int64_t t, int n_rows, int padded_rows,
    int hidden_chunks16, int hidden, int hidden_blocks, uint8_t* __restrict__ a_perm,
    float* __restrict__ a_scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t a_total = static_cast<int64_t>(padded_rows) * hidden_chunks16;
  int64_t scale_total = static_cast<int64_t>(padded_rows) * hidden_blocks;

  if (idx < a_total) {
    int pr = static_cast<int>(idx / hidden_chunks16);
    int chunk = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden_chunks16);
    uint4* dst = reinterpret_cast<uint4*>(a_perm + static_cast<int64_t>(pr) * hidden);
    if (pr < n_rows) {
      int tok = permuted_tok[pr];
      const uint4* src =
          reinterpret_cast<const uint4*>(hidden_fp8 + static_cast<int64_t>(tok) * hidden);
      dst[chunk] = src[chunk];
    } else {
      dst[chunk] = make_uint4(0, 0, 0, 0);
    }
  }

  if (idx < scale_total) {
    int pr = static_cast<int>(idx / hidden_blocks);
    int hb = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden_blocks);
    a_scale[static_cast<int64_t>(pr) * hidden_blocks + hb] =
        (pr < n_rows) ? hidden_scale[static_cast<int64_t>(hb) * t + permuted_tok[pr]] : 1.0f;
  }
}

__global__ void duplicate_all_expert_hidden_fp8_and_scale_kernel(
    const uint8_t* __restrict__ hidden_fp8, const float* __restrict__ hidden_scale, int64_t t,
    int padded_t, int hidden_chunks16, int hidden, int hidden_blocks,
    uint8_t* __restrict__ a_all, float* __restrict__ a_scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t rows = static_cast<int64_t>(32) * padded_t;
  int64_t a_total = rows * hidden_chunks16;
  int64_t scale_total = rows * hidden_blocks;

  if (idx < a_total) {
    int row = static_cast<int>(idx / hidden_chunks16);
    int chunk = static_cast<int>(idx - static_cast<int64_t>(row) * hidden_chunks16);
    int tok = row % padded_t;
    uint4* dst = reinterpret_cast<uint4*>(a_all + static_cast<int64_t>(row) * hidden);
    if (tok < t) {
      const uint4* src =
          reinterpret_cast<const uint4*>(hidden_fp8 + static_cast<int64_t>(tok) * hidden);
      dst[chunk] = src[chunk];
    } else {
      dst[chunk] = make_uint4(0, 0, 0, 0);
    }
  }

  if (idx < scale_total) {
    int row = static_cast<int>(idx / hidden_blocks);
    int hb = static_cast<int>(idx - static_cast<int64_t>(row) * hidden_blocks);
    int tok = row % padded_t;
    a_scale[static_cast<int64_t>(row) * hidden_blocks + hb] =
        (tok < t) ? hidden_scale[static_cast<int64_t>(hb) * t + tok] : 1.0f;
  }
}

__global__ void transpose_rowmajor_nk_to_colmajor_nk_kernel(const uint8_t* __restrict__ src,
                                                           int n, int k,
                                                           uint8_t* __restrict__ dst) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n) * k;
  if (idx >= total) return;
  int row_n = static_cast<int>(idx / k);
  int col_k = static_cast<int>(idx - static_cast<int64_t>(row_n) * k);
  dst[static_cast<int64_t>(col_k) * n + row_n] = src[idx];
}

__global__ void transpose_scale_nblock_kblock_to_kblock_nblock_kernel(
    const float* __restrict__ src, int n_blocks, int k_blocks, float* __restrict__ dst) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_blocks * k_blocks;
  if (idx >= total) return;
  int nb = idx / k_blocks;
  int kb = idx - nb * k_blocks;
  dst[kb * n_blocks + nb] = src[idx];
}

__global__ void copy_scale_nblock_kblock_kernel(const float* __restrict__ src, int total,
                                                float* __restrict__ dst) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  dst[idx] = src[idx];
}

__global__ void dequant_fp8_rows_kernel(const uint8_t* __restrict__ in_fp8,
                                        const float* __restrict__ in_scale, int rows, int cols,
                                        int row_scale_granularity, int col_blocks,
                                        bool scale_major_k,
                                        float* __restrict__ out_f32) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(rows) * cols;
  if (idx >= total) return;
  int row = static_cast<int>(idx / cols);
  int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  int rb = row / row_scale_granularity;
  int cb = col / 128;
  int row_scale_count = (rows + row_scale_granularity - 1) / row_scale_granularity;
  float scale = scale_major_k ? in_scale[static_cast<int64_t>(rb) * col_blocks + cb]
                              : in_scale[static_cast<int64_t>(cb) * row_scale_count + rb];
  out_f32[idx] = fp8_e4m3fn_to_float_device(in_fp8[idx]) * scale;
}

__global__ void dequant_fp8_rows_native_kernel(const uint8_t* __restrict__ in_fp8,
                                               const float* __restrict__ in_scale, int rows,
                                               int cols, int row_scale_granularity,
                                               int col_blocks, bool scale_major_k,
                                               float* __restrict__ out_f32) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(rows) * cols;
  if (idx >= total) return;
  int row = static_cast<int>(idx / cols);
  int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  int rb = row / row_scale_granularity;
  int cb = col / 128;
  int row_scale_count = (rows + row_scale_granularity - 1) / row_scale_granularity;
  float scale = scale_major_k ? in_scale[static_cast<int64_t>(rb) * col_blocks + cb]
                              : in_scale[static_cast<int64_t>(cb) * row_scale_count + rb];
  out_f32[idx] = fp8_native_to_float_device(in_fp8[idx]) * scale;
}

__global__ void dequant_fp8_rows_native_to_f16_kernel(const uint8_t* __restrict__ in_fp8,
                                                      const float* __restrict__ in_scale,
                                                      int rows, int cols,
                                                      int row_scale_granularity,
                                                      int col_blocks, bool scale_major_k,
                                                      uint16_t* __restrict__ out_f16) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(rows) * cols;
  if (idx >= total) return;
  int row = static_cast<int>(idx / cols);
  int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  int rb = row / row_scale_granularity;
  int cb = col / 128;
  int row_scale_count = (rows + row_scale_granularity - 1) / row_scale_granularity;
  float scale = scale_major_k ? in_scale[static_cast<int64_t>(rb) * col_blocks + cb]
                              : in_scale[static_cast<int64_t>(cb) * row_scale_count + rb];
  union {
    uint16_t u;
    __half h;
  } v;
  v.h = __float2half(fp8_e4m3fn_to_float_device(in_fp8[idx]) * scale);
  out_f16[idx] = v.u;
}

__global__ void dequant_fp8_experts_native_to_f16_kernel(const uint8_t* __restrict__ in_fp8,
                                                         const float* __restrict__ in_scale,
                                                         int experts, int rows, int cols,
                                                         int row_scale_granularity,
                                                         int col_blocks,
                                                         uint16_t* __restrict__ out_f16) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t matrix_elems = static_cast<int64_t>(rows) * cols;
  int64_t total = static_cast<int64_t>(experts) * matrix_elems;
  if (idx >= total) return;
  int expert = static_cast<int>(idx / matrix_elems);
  int64_t local = idx - static_cast<int64_t>(expert) * matrix_elems;
  int row = static_cast<int>(local / cols);
  int col = static_cast<int>(local - static_cast<int64_t>(row) * cols);
  int rb = row / row_scale_granularity;
  int cb = col / 128;
  int64_t scale_elems = static_cast<int64_t>((rows + row_scale_granularity - 1) /
                                             row_scale_granularity) *
                        col_blocks;
  float scale = in_scale[static_cast<int64_t>(expert) * scale_elems +
                         static_cast<int64_t>(rb) * col_blocks + cb];
  union {
    uint16_t u;
    __half h;
  } v;
  v.h = __float2half(fp8_e4m3fn_to_float_device(in_fp8[idx]) * scale);
  out_f16[idx] = v.u;
}

__global__ void dequant_fp8_rows_native_to_bf16_kernel(const uint8_t* __restrict__ in_fp8,
                                                       const float* __restrict__ in_scale,
                                                       int rows, int cols,
                                                       int row_scale_granularity,
                                                       int col_blocks, bool scale_major_k,
                                                       uint16_t* __restrict__ out_bf16) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(rows) * cols;
  if (idx >= total) return;
  int row = static_cast<int>(idx / cols);
  int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  int rb = row / row_scale_granularity;
  int cb = col / 128;
  int row_scale_count = (rows + row_scale_granularity - 1) / row_scale_granularity;
  float scale = scale_major_k ? in_scale[static_cast<int64_t>(rb) * col_blocks + cb]
                              : in_scale[static_cast<int64_t>(cb) * row_scale_count + rb];
  union {
    __nv_bfloat16 bf16;
    uint16_t u;
  } v;
  v.bf16 = __float2bfloat16_rn(fp8_e4m3fn_to_float_device(in_fp8[idx]) * scale);
  out_bf16[idx] = v.u;
}

__global__ void swiglu_quantize_bf16_to_fp8_kernel(const uint16_t* __restrict__ g1_bf16,
                                                   int intermediate, int n_rows, int padded_rows,
                                                   uint8_t* __restrict__ c_fp8,
                                                   float* __restrict__ c_scale,
                                                   bool scale_major_k) {
  int row = blockIdx.x;
  int ib = blockIdx.y;
  int tid = threadIdx.x;
  if (row >= padded_rows) return;

  extern __shared__ float smem[];
  float v = 0.0f;
  int i = ib * 128 + tid;
  if (tid < 128 && i < intermediate && row < n_rows) {
    const uint16_t* g1_row = g1_bf16 + static_cast<int64_t>(row) * (2 * intermediate);
    float x1 = bf16_to_float_device(g1_row[i]);
    float x2 = bf16_to_float_device(g1_row[i + intermediate]);
    v = x1 * siluf_device(x2);
  }
  smem[tid] = fabsf(v);
  __syncthreads();

  for (int offset = 64; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    __syncthreads();
  }

  float scale = fmaxf(smem[0] / 448.0f, 1.0e-8f);
  if (tid == 0) {
    if (scale_major_k) {
      c_scale[static_cast<int64_t>(row) * (intermediate / 128) + ib] = scale;
    } else {
      c_scale[static_cast<int64_t>(ib) * padded_rows + row] = scale;
    }
  }
  if (tid < 128 && i < intermediate) {
    c_fp8[static_cast<int64_t>(row) * intermediate + i] =
        (row < n_rows) ? float_to_e4m3_device(v / scale) : 0;
  }
}

__global__ void swiglu_quantize_f16_to_fp8_kernel(const uint16_t* __restrict__ g1_f16,
                                                  int intermediate, int n_rows, int padded_rows,
                                                  uint8_t* __restrict__ c_fp8,
                                                  float* __restrict__ c_scale,
                                                  bool scale_major_k) {
  int row = blockIdx.x;
  int ib = blockIdx.y;
  int tid = threadIdx.x;
  if (row >= padded_rows) return;

  extern __shared__ float smem[];
  float v = 0.0f;
  int i = ib * 128 + tid;
  if (tid < 128 && i < intermediate && row < n_rows) {
    const uint16_t* g1_row = g1_f16 + static_cast<int64_t>(row) * (2 * intermediate);
    float x1 = f16_to_float_device(g1_row[i]);
    float x2 = f16_to_float_device(g1_row[i + intermediate]);
    v = x1 * siluf_device(x2);
  }
  smem[tid] = fabsf(v);
  __syncthreads();

  for (int offset = 64; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    __syncthreads();
  }

  float scale = fmaxf(smem[0] / 448.0f, 1.0e-8f);
  if (tid == 0) {
    if (scale_major_k) {
      c_scale[static_cast<int64_t>(row) * (intermediate / 128) + ib] = scale;
    } else {
      c_scale[static_cast<int64_t>(ib) * padded_rows + row] = scale;
    }
  }
  if (tid < 128 && i < intermediate) {
    c_fp8[static_cast<int64_t>(row) * intermediate + i] =
        (row < n_rows) ? float_to_e4m3_device(v / scale) : 0;
  }
}

__global__ void swiglu_quantize_float_to_fp8_kernel(const float* __restrict__ g1_f32,
                                                    int intermediate, int n_rows,
                                                    int padded_rows,
                                                    uint8_t* __restrict__ c_fp8,
                                                    float* __restrict__ c_scale,
                                                    bool scale_major_k) {
  int row = blockIdx.x;
  int ib = blockIdx.y;
  int tid = threadIdx.x;
  if (row >= padded_rows) return;

  extern __shared__ float smem[];
  float v = 0.0f;
  int i = ib * 128 + tid;
  if (tid < 128 && i < intermediate && row < n_rows) {
    const float* g1_row = g1_f32 + static_cast<int64_t>(row) * (2 * intermediate);
    float x1 = g1_row[i];
    float x2 = g1_row[i + intermediate];
    v = x1 * siluf_device(x2);
  }
  smem[tid] = fabsf(v);
  __syncthreads();

  for (int offset = 64; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    __syncthreads();
  }

  float scale = fmaxf(smem[0] / 448.0f, 1.0e-8f);
  if (tid == 0) {
    int col_blocks = intermediate / 128;
    if (scale_major_k) {
      c_scale[static_cast<int64_t>(row) * col_blocks + ib] = scale;
    } else {
      c_scale[static_cast<int64_t>(ib) * padded_rows + row] = scale;
    }
  }
  if (tid < 128 && i < intermediate) {
    c_fp8[static_cast<int64_t>(row) * intermediate + i] =
        (row < n_rows) ? float_to_e4m3_device(v / scale) : 0;
  }
}

template <int RowsPerCta>
__global__ void swiglu_quantize_float_to_fp8_rows_per_cta_kernel(
    const float* __restrict__ g1_f32, int intermediate, int n_rows, int padded_rows,
    uint8_t* __restrict__ c_fp8, float* __restrict__ c_scale, bool scale_major_k) {
  int row_base = blockIdx.x * RowsPerCta;
  int ib = blockIdx.y;
  int tid = threadIdx.x;
  if (row_base >= padded_rows) return;

  extern __shared__ float smem[];
  float v[RowsPerCta];
  int i = ib * 128 + tid;
#pragma unroll
  for (int r = 0; r < RowsPerCta; ++r) {
    int row = row_base + r;
    float val = 0.0f;
    if (tid < 128 && i < intermediate && row < n_rows) {
      const float* g1_row = g1_f32 + static_cast<int64_t>(row) * (2 * intermediate);
      float x1 = g1_row[i];
      float x2 = g1_row[i + intermediate];
      val = x1 * siluf_device(x2);
    }
    v[r] = val;
    smem[r * 128 + tid] = fabsf(val);
  }
  __syncthreads();

  for (int offset = 64; offset > 0; offset >>= 1) {
#pragma unroll
    for (int r = 0; r < RowsPerCta; ++r) {
      if (tid < offset) {
        smem[r * 128 + tid] = fmaxf(smem[r * 128 + tid], smem[r * 128 + tid + offset]);
      }
    }
    __syncthreads();
  }

  int col_blocks = intermediate / 128;
#pragma unroll
  for (int r = 0; r < RowsPerCta; ++r) {
    int row = row_base + r;
    if (row >= padded_rows) continue;
    float scale = fmaxf(smem[r * 128] / 448.0f, 1.0e-8f);
    if (tid == 0) {
      if (scale_major_k) {
        c_scale[static_cast<int64_t>(row) * col_blocks + ib] = scale;
      } else {
        c_scale[static_cast<int64_t>(ib) * padded_rows + row] = scale;
      }
    }
    if (tid < 128 && i < intermediate) {
      c_fp8[static_cast<int64_t>(row) * intermediate + i] =
          (row < n_rows) ? float_to_e4m3_device(v[r] / scale) : 0;
    }
  }
}

__device__ __forceinline__ float load_g1_middle_value(const float* ptr) {
  return *ptr;
}

__device__ __forceinline__ float load_g1_middle_value(const uint16_t* ptr) {
  return f16_to_float_device(*ptr);
}

template <bool Interleaved>
__device__ __forceinline__ int g1_gate_col(int col, int intermediate) {
  if constexpr (!Interleaved) {
    (void)intermediate;
    return col;
  } else {
    return (col / 128) * 256 + (col & 127);
  }
}

template <bool Interleaved>
__device__ __forceinline__ int g1_up_col(int col, int intermediate) {
  if constexpr (!Interleaved) {
    return col + intermediate;
  } else {
    return (col / 128) * 256 + 128 + (col & 127);
  }
}

template <typename G1Type, bool Interleaved>
__global__ void grouped_swiglu_quantize_to_fp8_kernel(
    const G1Type* __restrict__ g1, int intermediate, int padded_total_rows,
    const int* __restrict__ expert_offsets, const int* __restrict__ padded_offsets,
    uint8_t* __restrict__ c_fp8, float* __restrict__ c_scale) {
  int padded_row = blockIdx.x;
  int ib = blockIdx.y;
  int tid = threadIdx.x;
  if (padded_row >= padded_total_rows) return;

  __shared__ int compact_row_sh;
  __shared__ int valid_row_sh;
  if (tid == 0) {
    int le = 0;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      if (padded_row >= padded_offsets[i]) le = i;
    }
    int local_row = padded_row - padded_offsets[le];
    int n_rows = expert_offsets[le + 1] - expert_offsets[le];
    bool valid_row = local_row < n_rows;
    compact_row_sh = expert_offsets[le] + local_row;
    valid_row_sh = valid_row ? 1 : 0;
  }
  __syncthreads();
  bool valid_row = valid_row_sh != 0;
  int compact_row = compact_row_sh;

  extern __shared__ float smem[];
  float v = 0.0f;
  int i = ib * 128 + tid;
  if (tid < 128 && i < intermediate && valid_row) {
    const G1Type* g1_row = g1 + static_cast<int64_t>(compact_row) * (2 * intermediate);
    float x1 = load_g1_middle_value(g1_row + g1_gate_col<Interleaved>(i, intermediate));
    float x2 = load_g1_middle_value(g1_row + g1_up_col<Interleaved>(i, intermediate));
    v = x1 * siluf_device(x2);
  }
  float m = fabsf(v);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, offset));
  }
  if ((tid & 31) == 0) smem[tid >> 5] = m;
  __syncthreads();

  m = (tid < 4) ? smem[tid] : 0.0f;
  if (tid < 32) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, offset));
    }
    if (tid == 0) smem[0] = fmaxf(m / 448.0f, 1.0e-8f);
  }
  __syncthreads();

  float scale = smem[0];
  int col_blocks = intermediate / 128;
  if (tid == 0) {
    c_scale[static_cast<int64_t>(padded_row) * col_blocks + ib] = scale;
  }
  if (tid < 128 && i < intermediate) {
    c_fp8[static_cast<int64_t>(padded_row) * intermediate + i] =
        valid_row ? float_to_e4m3_device(v / scale) : 0;
  }
}

__global__ void zero_grouped_padding_activation_rows_kernel(
    const int* __restrict__ expert_offsets, const int* __restrict__ padded_offsets,
    int intermediate, int intermediate_blocks, uint8_t* __restrict__ c_fp8,
    float* __restrict__ c_scale) {
  int le = blockIdx.x;
  if (le >= 32) return;
  int valid_rows = expert_offsets[le + 1] - expert_offsets[le];
  int padded_begin = padded_offsets[le] + valid_rows;
  int padded_end = padded_offsets[le + 1];
  int pad_rows = padded_end - padded_begin;
  if (pad_rows <= 0) return;

  int64_t fp8_elems = static_cast<int64_t>(pad_rows) * intermediate;
  int64_t scale_elems = static_cast<int64_t>(pad_rows) * intermediate_blocks;
  int64_t total = fp8_elems > scale_elems ? fp8_elems : scale_elems;
  for (int64_t idx = static_cast<int64_t>(threadIdx.x); idx < total; idx += blockDim.x) {
    if (idx < fp8_elems) {
      int row = static_cast<int>(idx / intermediate);
      int col = static_cast<int>(idx - static_cast<int64_t>(row) * intermediate);
      c_fp8[static_cast<int64_t>(padded_begin + row) * intermediate + col] = 0;
    }
    if (idx < scale_elems) {
      int row = static_cast<int>(idx / intermediate_blocks);
      int col = static_cast<int>(idx - static_cast<int64_t>(row) * intermediate_blocks);
      c_scale[static_cast<int64_t>(padded_begin + row) * intermediate_blocks + col] = 0.0f;
    }
  }
}

template <typename G1Type, bool Interleaved>
__global__ void grouped_swiglu_quantize_to_fp8_mapped_kernel(
    const G1Type* __restrict__ g1, int intermediate, int padded_total_rows,
    const int* __restrict__ compact_rows, uint8_t* __restrict__ c_fp8,
    float* __restrict__ c_scale) {
  int padded_row = blockIdx.x;
  int ib = blockIdx.y;
  int tid = threadIdx.x;
  if (padded_row >= padded_total_rows) return;

  int compact_row = compact_rows[padded_row];
  bool valid_row = compact_row >= 0;

  extern __shared__ float smem[];
  float v = 0.0f;
  int i = ib * 128 + tid;
  if (tid < 128 && i < intermediate && valid_row) {
    const G1Type* g1_row = g1 + static_cast<int64_t>(compact_row) * (2 * intermediate);
    float x1 = load_g1_middle_value(g1_row + g1_gate_col<Interleaved>(i, intermediate));
    float x2 = load_g1_middle_value(g1_row + g1_up_col<Interleaved>(i, intermediate));
    v = x1 * siluf_device(x2);
  }
  float m = fabsf(v);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, offset));
  }
  if ((tid & 31) == 0) smem[tid >> 5] = m;
  __syncthreads();

  m = (tid < 4) ? smem[tid] : 0.0f;
  if (tid < 32) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, offset));
    }
    if (tid == 0) smem[0] = fmaxf(m / 448.0f, 1.0e-8f);
  }
  __syncthreads();

  float scale = smem[0];
  int col_blocks = intermediate / 128;
  if (tid == 0) {
    c_scale[static_cast<int64_t>(padded_row) * col_blocks + ib] = scale;
  }
  if (tid < 128 && i < intermediate) {
    c_fp8[static_cast<int64_t>(padded_row) * intermediate + i] =
        valid_row ? float_to_e4m3_device(v / scale) : 0;
  }
}

template <int RowsPerCta, typename G1Type, bool Interleaved>
__global__ void grouped_swiglu_quantize_to_fp8_rows_per_cta_kernel(
    const G1Type* __restrict__ g1, int intermediate, int padded_total_rows,
    const int* __restrict__ expert_offsets, const int* __restrict__ padded_offsets,
    uint8_t* __restrict__ c_fp8, float* __restrict__ c_scale) {
  int padded_row_base = blockIdx.x * RowsPerCta;
  int ib = blockIdx.y;
  int tid = threadIdx.x;
  if (padded_row_base >= padded_total_rows) return;

  __shared__ int compact_row_sh[RowsPerCta];
  __shared__ int valid_row_sh[RowsPerCta];
  if (tid == 0) {
#pragma unroll
    for (int r = 0; r < RowsPerCta; ++r) {
      int padded_row = padded_row_base + r;
      if (padded_row >= padded_total_rows) {
        compact_row_sh[r] = 0;
        valid_row_sh[r] = 0;
        continue;
      }
      int le = 0;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        if (padded_row >= padded_offsets[j]) le = j;
      }
      int local_row = padded_row - padded_offsets[le];
      int n_rows = expert_offsets[le + 1] - expert_offsets[le];
      compact_row_sh[r] = expert_offsets[le] + local_row;
      valid_row_sh[r] = (local_row < n_rows) ? 1 : 0;
    }
  }
  __syncthreads();

  extern __shared__ float smem[];
  float v[RowsPerCta];
  bool valid_row[RowsPerCta];
  int i = ib * 128 + tid;
#pragma unroll
  for (int r = 0; r < RowsPerCta; ++r) {
    int padded_row = padded_row_base + r;
    float val = 0.0f;
    bool valid = valid_row_sh[r] != 0;
    int compact_row = compact_row_sh[r];
    if (padded_row < padded_total_rows && tid < 128 && i < intermediate && valid) {
      const G1Type* g1_row = g1 + static_cast<int64_t>(compact_row) * (2 * intermediate);
      float x1 = load_g1_middle_value(g1_row + g1_gate_col<Interleaved>(i, intermediate));
      float x2 = load_g1_middle_value(g1_row + g1_up_col<Interleaved>(i, intermediate));
      val = x1 * siluf_device(x2);
    }
    v[r] = val;
    valid_row[r] = valid;
    smem[r * 128 + tid] = fabsf(val);
  }
  __syncthreads();

  for (int offset = 64; offset > 0; offset >>= 1) {
#pragma unroll
    for (int r = 0; r < RowsPerCta; ++r) {
      if (tid < offset) {
        smem[r * 128 + tid] = fmaxf(smem[r * 128 + tid], smem[r * 128 + tid + offset]);
      }
    }
    __syncthreads();
  }

  int col_blocks = intermediate / 128;
#pragma unroll
  for (int r = 0; r < RowsPerCta; ++r) {
    int padded_row = padded_row_base + r;
    if (padded_row >= padded_total_rows) continue;
    float scale = fmaxf(smem[r * 128] / 448.0f, 1.0e-8f);
    if (tid == 0) {
      c_scale[static_cast<int64_t>(padded_row) * col_blocks + ib] = scale;
    }
    if (tid < 128 && i < intermediate) {
      c_fp8[static_cast<int64_t>(padded_row) * intermediate + i] =
          valid_row[r] ? float_to_e4m3_device(v[r] / scale) : 0;
    }
  }
}

template <int ColBlocksPerCta, typename G1Type, bool Interleaved>
__global__ void grouped_swiglu_quantize_to_fp8_col_blocks_per_cta_kernel(
    const G1Type* __restrict__ g1, int intermediate, int padded_total_rows,
    const int* __restrict__ expert_offsets, const int* __restrict__ padded_offsets,
    uint8_t* __restrict__ c_fp8, float* __restrict__ c_scale) {
  int padded_row = blockIdx.x;
  int ib_base = blockIdx.y * ColBlocksPerCta;
  int tid = threadIdx.x;
  if (padded_row >= padded_total_rows) return;

  __shared__ int compact_row_sh;
  __shared__ int valid_row_sh;
  if (tid == 0) {
    int le = 0;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      if (padded_row >= padded_offsets[i]) le = i;
    }
    int local_row = padded_row - padded_offsets[le];
    int n_rows = expert_offsets[le + 1] - expert_offsets[le];
    compact_row_sh = expert_offsets[le] + local_row;
    valid_row_sh = (local_row < n_rows) ? 1 : 0;
  }
  __syncthreads();

  int cb = tid / 128;
  int lane = tid - cb * 128;
  int ib = ib_base + cb;
  int col_blocks = intermediate / 128;
  int col = ib * 128 + lane;
  bool valid = valid_row_sh != 0;

  extern __shared__ float smem[];
  float v = 0.0f;
  if (cb < ColBlocksPerCta && ib < col_blocks && col < intermediate && valid) {
    const G1Type* g1_row = g1 + static_cast<int64_t>(compact_row_sh) * (2 * intermediate);
    float x1 = load_g1_middle_value(g1_row + g1_gate_col<Interleaved>(col, intermediate));
    float x2 = load_g1_middle_value(g1_row + g1_up_col<Interleaved>(col, intermediate));
    v = x1 * siluf_device(x2);
  }
  smem[tid] = fabsf(v);
  __syncthreads();

  for (int offset = 64; offset > 0; offset >>= 1) {
    if (lane < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    __syncthreads();
  }

  if (cb < ColBlocksPerCta && ib < col_blocks) {
    float scale = fmaxf(smem[cb * 128] / 448.0f, 1.0e-8f);
    if (lane == 0) {
      c_scale[static_cast<int64_t>(padded_row) * col_blocks + ib] = scale;
    }
    if (col < intermediate) {
      c_fp8[static_cast<int64_t>(padded_row) * intermediate + col] =
          valid ? float_to_e4m3_device(v / scale) : 0;
    }
  }
}

__global__ void compute_block_scale_128x128_kernel(const float* __restrict__ in_f32, int cols,
                                                   int n_rows, int padded_rows,
                                                   int row_blocks, int col_blocks,
                                                   float* __restrict__ out_scale,
                                                   bool scale_major_k) {
  int rb = blockIdx.x;
  int cb = blockIdx.y;
  int tid = threadIdx.x;
  if (rb >= row_blocks || cb >= col_blocks) return;

  extern __shared__ float smem[];
  float thread_max = 0.0f;
  int row0 = rb * 128;
  int col0 = cb * 128;
  int elems = 128 * 128;
  for (int linear = tid; linear < elems; linear += blockDim.x) {
    int r = row0 + linear / 128;
    int c = col0 + linear % 128;
    float v = 0.0f;
    if (r < n_rows && c < cols) {
      v = in_f32[static_cast<int64_t>(r) * cols + c];
    }
    thread_max = fmaxf(thread_max, fabsf(v));
  }
  smem[tid] = thread_max;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    __syncthreads();
  }
  if (tid == 0) {
    float scale = fmaxf(smem[0] / 448.0f, 1.0e-8f);
    if (scale_major_k) {
      out_scale[static_cast<int64_t>(rb) * col_blocks + cb] = scale;
    } else {
      out_scale[static_cast<int64_t>(cb) * row_blocks + rb] = scale;
    }
  }
}

__global__ void quantize_float_blocks_128x128_to_fp8_kernel(const float* __restrict__ in_f32,
                                                            const float* __restrict__ in_scale,
                                                            int cols, int n_rows,
                                                            int padded_rows, int row_blocks,
                                                            int col_blocks,
                                                            uint8_t* __restrict__ out_fp8,
                                                            bool scale_major_k) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(padded_rows) * cols;
  if (idx >= total) return;
  int row = static_cast<int>(idx / cols);
  int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  int rb = row / 128;
  int cb = col / 128;
  float scale = scale_major_k ? in_scale[static_cast<int64_t>(rb) * col_blocks + cb]
                              : in_scale[static_cast<int64_t>(cb) * row_blocks + rb];
  float v = (row < n_rows) ? in_f32[idx] : 0.0f;
  out_fp8[idx] = (row < n_rows) ? float_to_e4m3_device(v / scale) : 0;
}

__global__ void scatter_float_weighted_kernel(const float* __restrict__ d_f32, int hidden,
                                              int n_rows, const int* __restrict__ permuted_tok,
                                              const float* __restrict__ permuted_w,
                                              float* __restrict__ out_acc) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * hidden;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / hidden);
  int h = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden);
  int tok = permuted_tok[pr];
  float w = permuted_w[pr];
  out_acc[static_cast<int64_t>(tok) * hidden + h] += w * d_f32[idx];
}

__global__ void scatter_float_weighted_row_scaled_kernel(
    const float* __restrict__ d_f32, const float* __restrict__ row_scale, int hidden, int n_rows,
    const int* __restrict__ permuted_tok, const float* __restrict__ permuted_w,
    float* __restrict__ out_acc) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * hidden;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / hidden);
  int h = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden);
  int tok = permuted_tok[pr];
  float w = permuted_w[pr] * row_scale[pr];
  out_acc[static_cast<int64_t>(tok) * hidden + h] += w * d_f32[idx];
}

__global__ void scatter_all_float_weighted_row_scaled_kernel(
    const float* __restrict__ d_f32, const float* __restrict__ row_scale, int hidden,
    int total_rows, const int* __restrict__ expert_offsets, const int* __restrict__ permuted_tok,
    const float* __restrict__ permuted_w, float* __restrict__ out_acc) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(total_rows) * hidden;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / hidden);
  int h = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden);
  int le = 0;
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    if (pr >= expert_offsets[i]) le = i;
  }
  int scratch_row = pr + 4 * le;
  int tok = permuted_tok[pr];
  float w = permuted_w[pr] * row_scale[scratch_row];
  atomicAdd(out_acc + static_cast<int64_t>(tok) * hidden + h,
            w * d_f32[static_cast<int64_t>(scratch_row) * hidden + h]);
}

__global__ void scatter_all_float_weighted_kernel(
    const float* __restrict__ d_f32, int hidden, int total_rows,
    const int* __restrict__ expert_offsets, const int* __restrict__ permuted_tok,
    const float* __restrict__ permuted_w, float* __restrict__ out_acc) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(total_rows) * hidden;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / hidden);
  int h = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden);
  int le = 0;
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    if (pr >= expert_offsets[i]) le = i;
  }
  int scratch_row = pr + 4 * le;
  int tok = permuted_tok[pr];
  float w = permuted_w[pr];
  atomicAdd(out_acc + static_cast<int64_t>(tok) * hidden + h,
            w * d_f32[static_cast<int64_t>(scratch_row) * hidden + h]);
}

__global__ void scatter_all_float_weighted_by_row_kernel(
    const float* __restrict__ d_f32, int hidden, int total_rows,
    const int* __restrict__ expert_offsets, const int* __restrict__ permuted_tok,
    const float* __restrict__ permuted_w, float* __restrict__ out_acc) {
  int pr = blockIdx.x;
  int h = blockIdx.y * blockDim.x + threadIdx.x;
  if (pr >= total_rows || h >= hidden) return;

  int le = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    if (pr >= expert_offsets[i]) le = i;
  }
  int scratch_row = pr + 4 * le;
  int tok = permuted_tok[pr];
  float w = permuted_w[pr];
  atomicAdd(out_acc + static_cast<int64_t>(tok) * hidden + h,
            w * d_f32[static_cast<int64_t>(scratch_row) * hidden + h]);
}

__global__ void scatter_all_float_weighted_by_expert_row_kernel(
    const float* __restrict__ d_f32, int hidden, int total_rows,
    const int* __restrict__ permuted_tok, const int* __restrict__ permuted_expert,
    const float* __restrict__ permuted_w, float* __restrict__ out_acc) {
  int pr = blockIdx.x;
  int h = blockIdx.y * blockDim.x + threadIdx.x;
  if (pr >= total_rows || h >= hidden) return;

  int scratch_row = pr + 4 * permuted_expert[pr];
  int tok = permuted_tok[pr];
  float w = permuted_w[pr];
  atomicAdd(out_acc + static_cast<int64_t>(tok) * hidden + h,
            w * d_f32[static_cast<int64_t>(scratch_row) * hidden + h]);
}

__global__ void scatter_all_float_weighted_by_padded_expert_row_kernel(
    const float* __restrict__ d_f32, int hidden, int total_rows,
    const int* __restrict__ expert_offsets, const int* __restrict__ padded_offsets,
    const int* __restrict__ permuted_tok, const int* __restrict__ permuted_expert,
    const float* __restrict__ permuted_w, float* __restrict__ out_acc) {
  int pr = blockIdx.x;
  int h = blockIdx.y * blockDim.x + threadIdx.x;
  if (pr >= total_rows || h >= hidden) return;

  int le = permuted_expert[pr];
  int scratch_row = padded_offsets[le] + (pr - expert_offsets[le]);
  int tok = permuted_tok[pr];
  float w = permuted_w[pr];
  atomicAdd(out_acc + static_cast<int64_t>(tok) * hidden + h,
            w * d_f32[static_cast<int64_t>(scratch_row) * hidden + h]);
}

__global__ void gather_scratch_topk_to_bf16_kernel(
    const float* __restrict__ d_f32, int hidden, int64_t t,
    const int* __restrict__ routed_positions, const int* __restrict__ routed_local_experts,
    const float* __restrict__ routed_weights, uint16_t* __restrict__ out_bf16) {
  int tok = blockIdx.x;
  int h = blockIdx.y * blockDim.x + threadIdx.x;
  if (tok >= t || h >= hidden) return;

  __shared__ int smem_pos[8];
  __shared__ int smem_le[8];
  __shared__ float smem_w[8];
  const int base = tok * 8;
  if (threadIdx.x < 8) {
    smem_pos[threadIdx.x] = routed_positions[base + threadIdx.x];
    smem_le[threadIdx.x] = routed_local_experts[base + threadIdx.x];
    smem_w[threadIdx.x] = routed_weights[base + threadIdx.x];
  }
  __syncthreads();

  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int pos = smem_pos[i];
    if (pos < 0) break;
    int le = smem_le[i];
    float w = smem_w[i];
    int scratch_row = pos + 4 * le;
    acc += w * d_f32[static_cast<int64_t>(scratch_row) * hidden + h];
  }
  union {
    __nv_bfloat16 bf16;
    uint16_t u;
  } packed;
  packed.bf16 = __float2bfloat16_rn(acc);
  out_bf16[static_cast<int64_t>(tok) * hidden + h] = packed.u;
}

__device__ __forceinline__ uint16_t float_to_bf16_bits_device(float x) {
  union {
    __nv_bfloat16 bf16;
    uint16_t u;
  } packed;
  packed.bf16 = __float2bfloat16_rn(x);
  return packed.u;
}

template <int Vec>
__device__ __forceinline__ void store_bf16_vector_device(uint16_t* dst, const float (&acc)[Vec],
                                                        int h0, int hidden) {
#pragma unroll
  for (int v = 0; v < Vec; ++v) {
    int h = h0 + v;
    if (h < hidden) dst[v] = float_to_bf16_bits_device(acc[v]);
  }
}

template <>
__device__ __forceinline__ void store_bf16_vector_device<2>(uint16_t* dst,
                                                            const float (&acc)[2], int h0,
                                                            int hidden) {
  if (h0 + 2 <= hidden) {
    uint32_t packed = static_cast<uint32_t>(float_to_bf16_bits_device(acc[0])) |
                      (static_cast<uint32_t>(float_to_bf16_bits_device(acc[1])) << 16);
    *reinterpret_cast<uint32_t*>(dst) = packed;
  } else {
    if (h0 < hidden) dst[0] = float_to_bf16_bits_device(acc[0]);
  }
}

template <>
__device__ __forceinline__ void store_bf16_vector_device<4>(uint16_t* dst,
                                                            const float (&acc)[4], int h0,
                                                            int hidden) {
  if (h0 + 4 <= hidden) {
    uint2 packed;
    packed.x = static_cast<uint32_t>(float_to_bf16_bits_device(acc[0])) |
               (static_cast<uint32_t>(float_to_bf16_bits_device(acc[1])) << 16);
    packed.y = static_cast<uint32_t>(float_to_bf16_bits_device(acc[2])) |
               (static_cast<uint32_t>(float_to_bf16_bits_device(acc[3])) << 16);
    *reinterpret_cast<uint2*>(dst) = packed;
  } else {
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      int h = h0 + v;
      if (h < hidden) dst[v] = float_to_bf16_bits_device(acc[v]);
    }
  }
}

template <>
__device__ __forceinline__ void store_bf16_vector_device<8>(uint16_t* dst,
                                                            const float (&acc)[8], int h0,
                                                            int hidden) {
  if (h0 + 8 <= hidden) {
    uint2 packed0;
    uint2 packed1;
    packed0.x = static_cast<uint32_t>(float_to_bf16_bits_device(acc[0])) |
                (static_cast<uint32_t>(float_to_bf16_bits_device(acc[1])) << 16);
    packed0.y = static_cast<uint32_t>(float_to_bf16_bits_device(acc[2])) |
                (static_cast<uint32_t>(float_to_bf16_bits_device(acc[3])) << 16);
    packed1.x = static_cast<uint32_t>(float_to_bf16_bits_device(acc[4])) |
                (static_cast<uint32_t>(float_to_bf16_bits_device(acc[5])) << 16);
    packed1.y = static_cast<uint32_t>(float_to_bf16_bits_device(acc[6])) |
                (static_cast<uint32_t>(float_to_bf16_bits_device(acc[7])) << 16);
    reinterpret_cast<uint2*>(dst)[0] = packed0;
    reinterpret_cast<uint2*>(dst)[1] = packed1;
  } else {
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      int h = h0 + v;
      if (h < hidden) dst[v] = float_to_bf16_bits_device(acc[v]);
    }
  }
}

template <int Vec>
__device__ __forceinline__ void store_bf16_vector_exact_device(uint16_t* dst,
                                                               const float (&acc)[Vec]) {
#pragma unroll
  for (int v = 0; v < Vec; ++v) {
    dst[v] = float_to_bf16_bits_device(acc[v]);
  }
}

template <>
__device__ __forceinline__ void store_bf16_vector_exact_device<2>(uint16_t* dst,
                                                                  const float (&acc)[2]) {
  uint32_t packed = static_cast<uint32_t>(float_to_bf16_bits_device(acc[0])) |
                    (static_cast<uint32_t>(float_to_bf16_bits_device(acc[1])) << 16);
  *reinterpret_cast<uint32_t*>(dst) = packed;
}

template <>
__device__ __forceinline__ void store_bf16_vector_exact_device<4>(uint16_t* dst,
                                                                  const float (&acc)[4]) {
  uint2 packed;
  packed.x = static_cast<uint32_t>(float_to_bf16_bits_device(acc[0])) |
             (static_cast<uint32_t>(float_to_bf16_bits_device(acc[1])) << 16);
  packed.y = static_cast<uint32_t>(float_to_bf16_bits_device(acc[2])) |
             (static_cast<uint32_t>(float_to_bf16_bits_device(acc[3])) << 16);
  *reinterpret_cast<uint2*>(dst) = packed;
}

template <>
__device__ __forceinline__ void store_bf16_vector_exact_device<8>(uint16_t* dst,
                                                                  const float (&acc)[8]) {
  uint2 packed0;
  uint2 packed1;
  packed0.x = static_cast<uint32_t>(float_to_bf16_bits_device(acc[0])) |
              (static_cast<uint32_t>(float_to_bf16_bits_device(acc[1])) << 16);
  packed0.y = static_cast<uint32_t>(float_to_bf16_bits_device(acc[2])) |
              (static_cast<uint32_t>(float_to_bf16_bits_device(acc[3])) << 16);
  packed1.x = static_cast<uint32_t>(float_to_bf16_bits_device(acc[4])) |
              (static_cast<uint32_t>(float_to_bf16_bits_device(acc[5])) << 16);
  packed1.y = static_cast<uint32_t>(float_to_bf16_bits_device(acc[6])) |
              (static_cast<uint32_t>(float_to_bf16_bits_device(acc[7])) << 16);
  reinterpret_cast<uint2*>(dst)[0] = packed0;
  reinterpret_cast<uint2*>(dst)[1] = packed1;
}

template <int Vec, bool Exact = false>
__global__ void gather_scratch_topk_to_bf16_vec_kernel(
    const float* __restrict__ d_f32, int hidden, int64_t t,
    const int* __restrict__ routed_positions, const int* __restrict__ routed_local_experts,
    const float* __restrict__ routed_weights, uint16_t* __restrict__ out_bf16) {
  int tok = blockIdx.x;
  int h0 = (blockIdx.y * blockDim.x + threadIdx.x) * Vec;
  if (tok >= t || h0 >= hidden) return;

  __shared__ int smem_pos[8];
  __shared__ int smem_le[8];
  __shared__ float smem_w[8];
  const int base = tok * 8;
  if (threadIdx.x < 8) {
    smem_pos[threadIdx.x] = routed_positions[base + threadIdx.x];
    smem_le[threadIdx.x] = routed_local_experts[base + threadIdx.x];
    smem_w[threadIdx.x] = routed_weights[base + threadIdx.x];
  }
  __syncthreads();

  float acc[Vec];
#pragma unroll
  for (int v = 0; v < Vec; ++v) acc[v] = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int pos = smem_pos[i];
    if (pos < 0) break;
    int le = smem_le[i];
    float w = smem_w[i];
    const float* src = d_f32 + static_cast<int64_t>(pos + 4 * le) * hidden + h0;
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      int h = h0 + v;
      if constexpr (Exact) {
        (void)h;
        acc[v] += w * src[v];
      } else {
        if (h < hidden) acc[v] += w * src[v];
      }
    }
  }
  uint16_t* dst = out_bf16 + static_cast<int64_t>(tok) * hidden + h0;
  if constexpr (Exact) {
    store_bf16_vector_exact_device<Vec>(dst, acc);
  } else {
    store_bf16_vector_device<Vec>(dst, acc, h0, hidden);
  }
}

__global__ void gather_padded_scratch_topk_to_bf16_kernel(
    const float* __restrict__ d_f32, int hidden, int64_t t,
    const int* __restrict__ expert_offsets, const int* __restrict__ padded_offsets,
    const int* __restrict__ routed_positions, const int* __restrict__ routed_local_experts,
    const float* __restrict__ routed_weights, uint16_t* __restrict__ out_bf16) {
  int tok = blockIdx.x;
  int h = blockIdx.y * blockDim.x + threadIdx.x;
  if (tok >= t || h >= hidden) return;

  __shared__ int smem_pos[8];
  __shared__ int smem_le[8];
  __shared__ float smem_w[8];
  const int base = tok * 8;
  if (threadIdx.x < 8) {
    smem_pos[threadIdx.x] = routed_positions[base + threadIdx.x];
    smem_le[threadIdx.x] = routed_local_experts[base + threadIdx.x];
    smem_w[threadIdx.x] = routed_weights[base + threadIdx.x];
  }
  __syncthreads();

  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int pos = smem_pos[i];
    if (pos < 0) break;
    int le = smem_le[i];
    int scratch_row = padded_offsets[le] + (pos - expert_offsets[le]);
    acc += smem_w[i] * d_f32[static_cast<int64_t>(scratch_row) * hidden + h];
  }
  union {
    __nv_bfloat16 bf16;
    uint16_t u;
  } packed;
  packed.bf16 = __float2bfloat16_rn(acc);
  out_bf16[static_cast<int64_t>(tok) * hidden + h] = packed.u;
}

template <int Vec, bool Exact = false>
__global__ void gather_padded_scratch_topk_to_bf16_vec_kernel(
    const float* __restrict__ d_f32, int hidden, int64_t t,
    const int* __restrict__ expert_offsets, const int* __restrict__ padded_offsets,
    const int* __restrict__ routed_positions, const int* __restrict__ routed_local_experts,
    const float* __restrict__ routed_weights, uint16_t* __restrict__ out_bf16) {
  int tok = blockIdx.x;
  int h0 = (blockIdx.y * blockDim.x + threadIdx.x) * Vec;
  if (tok >= t || h0 >= hidden) return;

  __shared__ int smem_pos[8];
  __shared__ int smem_le[8];
  __shared__ float smem_w[8];
  const int base = tok * 8;
  if (threadIdx.x < 8) {
    smem_pos[threadIdx.x] = routed_positions[base + threadIdx.x];
    smem_le[threadIdx.x] = routed_local_experts[base + threadIdx.x];
    smem_w[threadIdx.x] = routed_weights[base + threadIdx.x];
  }
  __syncthreads();

  float acc[Vec];
#pragma unroll
  for (int v = 0; v < Vec; ++v) acc[v] = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int pos = smem_pos[i];
    if (pos < 0) break;
    int le = smem_le[i];
    int scratch_row = padded_offsets[le] + (pos - expert_offsets[le]);
    const float* src = d_f32 + static_cast<int64_t>(scratch_row) * hidden + h0;
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      int h = h0 + v;
      if constexpr (Exact) {
        (void)h;
        acc[v] += smem_w[i] * src[v];
      } else {
        if (h < hidden) acc[v] += smem_w[i] * src[v];
      }
    }
  }
  uint16_t* dst = out_bf16 + static_cast<int64_t>(tok) * hidden + h0;
  if constexpr (Exact) {
    store_bf16_vector_exact_device<Vec>(dst, acc);
  } else {
    store_bf16_vector_device<Vec>(dst, acc, h0, hidden);
  }
}

template <int Vec>
__global__ void gather_all_expert_scratch_topk_to_bf16_vec_kernel(
    const float* __restrict__ d_f32, int hidden, int64_t t, int padded_t,
    const int* __restrict__ padded_offsets, const int* __restrict__ routed_local_experts,
    const float* __restrict__ routed_weights, uint16_t* __restrict__ out_bf16) {
  int tok = blockIdx.x;
  int h0 = (blockIdx.y * blockDim.x + threadIdx.x) * Vec;
  if (tok >= t || h0 >= hidden) return;

  __shared__ int smem_le[8];
  __shared__ float smem_w[8];
  const int base = tok * 8;
  if (threadIdx.x < 8) {
    smem_le[threadIdx.x] = routed_local_experts[base + threadIdx.x];
    smem_w[threadIdx.x] = routed_weights[base + threadIdx.x];
  }
  __syncthreads();

  float acc[Vec];
#pragma unroll
  for (int v = 0; v < Vec; ++v) acc[v] = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int le = smem_le[i];
    if (le < 0) break;
    float w = smem_w[i];
    const float* src =
        d_f32 + static_cast<int64_t>(padded_offsets[le] + tok) * hidden + h0;
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      int h = h0 + v;
      if (h < hidden) acc[v] += w * src[v];
    }
  }
  uint16_t* dst = out_bf16 + static_cast<int64_t>(tok) * hidden + h0;
  store_bf16_vector_device<Vec>(dst, acc, h0, hidden);
}

__global__ void bf16_matrix_to_float_kernel(const uint16_t* __restrict__ in, int64_t n,
                                           float* __restrict__ out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = bf16_to_float_device(in[idx]);
}

__global__ void f16_matrix_to_float_kernel(const uint16_t* __restrict__ in, int64_t n,
                                          float* __restrict__ out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = f16_to_float_device(in[idx]);
}

__global__ void f16_rows_scaled_to_float_kernel(const uint16_t* __restrict__ in,
                                                const float* __restrict__ row_scale, int cols,
                                                int n_rows, int padded_rows,
                                                float* __restrict__ out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(padded_rows) * cols;
  if (idx >= total) return;
  int row = static_cast<int>(idx / cols);
  float v = (row < n_rows) ? f16_to_float_device(in[idx]) * row_scale[row] : 0.0f;
  out[idx] = v;
}

__global__ void float_to_f16_kernel(const float* __restrict__ in, int64_t n,
                                    uint16_t* __restrict__ out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  union {
    uint16_t u;
    __half h;
  } v;
  v.h = __float2half(in[idx]);
  out[idx] = v.u;
}

__global__ void float_rows_scaled_to_f16_kernel(const float* __restrict__ in, int cols,
                                                int n_rows, int padded_rows,
                                                uint16_t* __restrict__ out,
                                                float* __restrict__ row_scale) {
  int row = blockIdx.x;
  if (row >= padded_rows) return;
  __shared__ float smem[256];
  float max_abs = 0.0f;
  if (row < n_rows) {
    const float* in_row = in + static_cast<int64_t>(row) * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
      max_abs = fmaxf(max_abs, fabsf(in_row[col]));
    }
  }
  smem[threadIdx.x] = max_abs;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float scale = fmaxf(smem[0] / 60000.0f, 1.0f);
  if (threadIdx.x == 0) row_scale[row] = scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float v = (row < n_rows) ? in[static_cast<int64_t>(row) * cols + col] / scale : 0.0f;
    union {
      __half h;
      uint16_t u;
    } packed;
    packed.h = __float2half(v);
    out[static_cast<int64_t>(row) * cols + col] = packed.u;
  }
}

__global__ void swiglu_round_g1_rows_scaled_to_f16_kernel(const float* __restrict__ g1,
                                                          int intermediate, int n_rows,
                                                          int padded_rows,
                                                          uint16_t* __restrict__ out,
                                                          float* __restrict__ row_scale) {
  int row = blockIdx.x;
  if (row >= padded_rows) return;
  __shared__ float smem[256];
  float max_abs = 0.0f;
  if (row < n_rows) {
    const float* g1_row = g1 + static_cast<int64_t>(row) * (2 * intermediate);
    for (int col = threadIdx.x; col < intermediate; col += blockDim.x) {
      float x1 = __half2float(__float2half(g1_row[col]));
      float x2 = __half2float(__float2half(g1_row[col + intermediate]));
      float y = x1 * siluf_device(x2);
      max_abs = fmaxf(max_abs, fabsf(y));
    }
  }
  smem[threadIdx.x] = max_abs;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float scale = fmaxf(smem[0] / 60000.0f, 1.0f);
  if (threadIdx.x == 0) row_scale[row] = scale;
  for (int col = threadIdx.x; col < intermediate; col += blockDim.x) {
    float v = 0.0f;
    if (row < n_rows) {
      const float* g1_row = g1 + static_cast<int64_t>(row) * (2 * intermediate);
      float x1 = __half2float(__float2half(g1_row[col]));
      float x2 = __half2float(__float2half(g1_row[col + intermediate]));
      v = (x1 * siluf_device(x2)) / scale;
    }
    union {
      __half h;
      uint16_t u;
    } packed;
    packed.h = __float2half(v);
    out[static_cast<int64_t>(row) * intermediate + col] = packed.u;
  }
}

__global__ void swiglu_g1_f16_rows_scaled_to_f16_kernel(const uint16_t* __restrict__ g1,
                                                        int intermediate, int n_rows,
                                                        int padded_rows,
                                                        uint16_t* __restrict__ out,
                                                        float* __restrict__ row_scale) {
  int row = blockIdx.x;
  if (row >= padded_rows) return;
  __shared__ float smem[256];
  float max_abs = 0.0f;
  if (row < n_rows) {
    const uint16_t* g1_row = g1 + static_cast<int64_t>(row) * (2 * intermediate);
    for (int col = threadIdx.x; col < intermediate; col += blockDim.x) {
      float x1 = f16_to_float_device(g1_row[col]);
      float x2 = f16_to_float_device(g1_row[col + intermediate]);
      float y = x1 * siluf_device(x2);
      max_abs = fmaxf(max_abs, fabsf(y));
    }
  }
  smem[threadIdx.x] = max_abs;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float scale = fmaxf(smem[0] / 60000.0f, 1.0f);
  if (threadIdx.x == 0) row_scale[row] = scale;
  for (int col = threadIdx.x; col < intermediate; col += blockDim.x) {
    float v = 0.0f;
    if (row < n_rows) {
      const uint16_t* g1_row = g1 + static_cast<int64_t>(row) * (2 * intermediate);
      float x1 = f16_to_float_device(g1_row[col]);
      float x2 = f16_to_float_device(g1_row[col + intermediate]);
      v = (x1 * siluf_device(x2)) / scale;
    }
    union {
      __half h;
      uint16_t u;
    } packed;
    packed.h = __float2half(v);
    out[static_cast<int64_t>(row) * intermediate + col] = packed.u;
  }
}

__global__ void float_to_bf16_kernel(const float* __restrict__ in, int64_t n,
                                     uint16_t* __restrict__ out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  union {
    __nv_bfloat16 bf16;
    uint16_t u;
  } v;
  v.bf16 = __float2bfloat16_rn(in[idx]);
  out[idx] = v.u;
}

__global__ void compare_abs_diff_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                        int64_t n, float* __restrict__ max_out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float diff = fabsf(a[idx] - b[idx]);
  atomicMax(reinterpret_cast<int*>(max_out), __float_as_int(diff));
}

__global__ void compare_g1_stats_kernel(const float* __restrict__ cur,
                                        const float* __restrict__ ref,
                                        int64_t n, float* __restrict__ max_out,
                                        int* __restrict__ bad_count,
                                        float* __restrict__ samples) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float a = cur[idx];
  float b = ref[idx];
  if (idx < 16) {
    samples[idx] = a;
    samples[16 + idx] = b;
  }
  if (!isfinite(a) || !isfinite(b)) {
    atomicAdd(bad_count, 1);
    return;
  }
  float diff = fabsf(a - b);
  atomicMax(reinterpret_cast<int*>(max_out), __float_as_int(diff));
}

}  // namespace

DeviceMxfpGemmModule::DeviceMxfpGemmModule(int hidden, int intermediate, int block)
    : hidden_(hidden),
      intermediate_(intermediate),
      block_(block),
      gemm1_out_(2 * intermediate),
      hidden_blocks_(hidden / block),
      intermediate_blocks_(intermediate / block),
      gemm1_out_blocks_(gemm1_out_ / block),
      max_t_(0),
      emulate_fp8_unit_(false),
      emulate_fp16_operands_(false),
      emulate_acc_half_(false),
      tc_fp16_middle_(false),
      g1_dev_(nullptr),
      c_dev_(nullptr),
      tc_max_rows_(0),
      tc_path_enabled_(FIB_HAS_DIRECT_CUTLASS_SM100),
      tc_a_fp8_dev_(nullptr),
      tc_b_col_dev_(nullptr),
      tc_a_scale_dev_(nullptr),
      tc_b_scale_dev_(nullptr),
      tc_g1_f32_dev_(nullptr),
      tc_c_fp8_dev_(nullptr),
      tc_c_scale_dev_(nullptr),
      tc_d_f32_dev_(nullptr),
      tc_m_indptr_dev_(nullptr),
      tc_m_indptr_host_(nullptr),
      tc_padded_compact_rows_dev_(nullptr),
      tc_int_workspace_dev_(nullptr),
      tc_float_workspace_dev_(nullptr),
      tc_group_int_workspace_dev_(nullptr),
      tc_group_float_workspace_dev_(nullptr) {
  const char* env = std::getenv("FIB_EMULATE_FP8_UNIT");
  emulate_fp8_unit_ = (env != nullptr && env[0] == '1');
  const char* env_fp16_op = std::getenv("FIB_EMULATE_FP16_OPERANDS");
  emulate_fp16_operands_ = (env_fp16_op != nullptr && env_fp16_op[0] == '1');
  const char* env_acc = std::getenv("FIB_EMULATE_FP8_ACC_HALF");
  emulate_acc_half_ = (env_acc != nullptr && env_acc[0] == '1');
  const char* env_fp16_middle = std::getenv("FIB_MOE_TC_FP16_MIDDLE");
  tc_fp16_middle_ = (env_fp16_middle != nullptr && env_fp16_middle[0] == '1');
  if (emulate_fp8_unit_) {
    std::fprintf(stderr,
                 "[mxfp] FIB_EMULATE_FP8_UNIT=1 (TC-like emulation: FP8-like operands, FP32 accumulate)\n");
  }
  if (emulate_acc_half_) {
    std::fprintf(stderr, "[mxfp] FIB_EMULATE_FP8_ACC_HALF=1 (half accumulate enabled)\n");
  }
  if (emulate_fp16_operands_) {
    std::fprintf(stderr, "[mxfp] FIB_EMULATE_FP16_OPERANDS=1 (fp16*fp16 multiply emulation enabled)\n");
  }
  if (tc_fp16_middle_) {
    std::fprintf(stderr,
                 "[mxfp] FIB_MOE_TC_FP16_MIDDLE=1 (experimental fp16 middle path enabled)\n");
  }
  const char* env_tc = std::getenv("FIB_MOE_TC");
  if (env_tc != nullptr) {
    tc_path_enabled_ = (env_tc[0] != '0');
  }
  const char* env_no_tc = std::getenv("FIB_MOE_NO_TC");
  if (env_no_tc != nullptr && env_no_tc[0] == '1') {
    tc_path_enabled_ = false;
  }
  if (tc_path_enabled_ || env_tc != nullptr || env_no_tc != nullptr) {
    std::fprintf(stderr,
                 "[mxfp] CUTLASS expert GEMM path %s (%s)\n",
                 tc_path_enabled_ ? "enabled" : "disabled",
                 FIB_HAS_DIRECT_CUTLASS_SM100 ? "direct CUTLASS available"
                                              : "direct CUTLASS unavailable");
  }
  if (emulate_fp8_unit_ || emulate_fp16_operands_ || emulate_acc_half_ || tc_fp16_middle_ ||
      env_tc != nullptr || env_no_tc != nullptr || FIB_HAS_DIRECT_CUTLASS_SM100) {
    std::fflush(stderr);
  }
}

DeviceMxfpGemmModule::~DeviceMxfpGemmModule() {
  if (g1_dev_ != nullptr) cudaFree(g1_dev_);
  if (c_dev_ != nullptr) cudaFree(c_dev_);
  if (tc_a_fp8_dev_ != nullptr) cudaFree(tc_a_fp8_dev_);
  if (tc_b_col_dev_ != nullptr) cudaFree(tc_b_col_dev_);
  if (tc_a_scale_dev_ != nullptr) cudaFree(tc_a_scale_dev_);
  if (tc_b_scale_dev_ != nullptr) cudaFree(tc_b_scale_dev_);
  if (tc_g1_f32_dev_ != nullptr) cudaFree(tc_g1_f32_dev_);
  if (tc_c_fp8_dev_ != nullptr) cudaFree(tc_c_fp8_dev_);
  if (tc_c_scale_dev_ != nullptr) cudaFree(tc_c_scale_dev_);
  if (tc_d_f32_dev_ != nullptr) cudaFree(tc_d_f32_dev_);
  if (tc_m_indptr_dev_ != nullptr) cudaFree(tc_m_indptr_dev_);
  if (tc_m_indptr_host_ != nullptr) cudaFreeHost(tc_m_indptr_host_);
  if (tc_padded_compact_rows_dev_ != nullptr) cudaFree(tc_padded_compact_rows_dev_);
  if (tc_int_workspace_dev_ != nullptr) cudaFree(tc_int_workspace_dev_);
  if (tc_float_workspace_dev_ != nullptr) cudaFree(tc_float_workspace_dev_);
  if (tc_group_int_workspace_dev_ != nullptr) cudaFree(tc_group_int_workspace_dev_);
  if (tc_group_float_workspace_dev_ != nullptr) cudaFree(tc_group_float_workspace_dev_);
}

void DeviceMxfpGemmModule::EnsureWorkspace(int64_t t, cudaStream_t stream) {
  (void)stream;
  if (t <= max_t_ && g1_dev_ != nullptr && c_dev_ != nullptr) return;

  if (g1_dev_ != nullptr) {
    cudaFree(g1_dev_);
    g1_dev_ = nullptr;
  }
  if (c_dev_ != nullptr) {
    cudaFree(c_dev_);
    c_dev_ = nullptr;
  }

  max_t_ = t;
  cudaError_t e1 = cudaMalloc(&g1_dev_, static_cast<size_t>(t) * static_cast<size_t>(gemm1_out_) * sizeof(float));
  cudaError_t e2 = cudaMalloc(&c_dev_, static_cast<size_t>(t) * static_cast<size_t>(intermediate_) * sizeof(float));
  if (e1 != cudaSuccess || e2 != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed for DeviceMxfpGemmModule workspace");
  }
}

bool DeviceMxfpGemmModule::SupportsTcPath() const {
  return tc_path_enabled_ && FIB_HAS_DIRECT_CUTLASS_SM100;
}

void DeviceMxfpGemmModule::EnsureTcWorkspace(int rows) {
  int padded_rows = (rows + 3) & ~3;
  if (padded_rows <= tc_max_rows_ && tc_a_fp8_dev_ != nullptr) return;

  if (tc_a_fp8_dev_ != nullptr) cudaFree(tc_a_fp8_dev_);
  if (tc_b_col_dev_ != nullptr) cudaFree(tc_b_col_dev_);
  if (tc_a_scale_dev_ != nullptr) cudaFree(tc_a_scale_dev_);
  if (tc_b_scale_dev_ != nullptr) cudaFree(tc_b_scale_dev_);
  if (tc_g1_f32_dev_ != nullptr) cudaFree(tc_g1_f32_dev_);
  if (tc_c_fp8_dev_ != nullptr) cudaFree(tc_c_fp8_dev_);
  if (tc_c_scale_dev_ != nullptr) cudaFree(tc_c_scale_dev_);
  if (tc_d_f32_dev_ != nullptr) cudaFree(tc_d_f32_dev_);
  if (tc_m_indptr_dev_ != nullptr) cudaFree(tc_m_indptr_dev_);
  if (tc_padded_compact_rows_dev_ != nullptr) cudaFree(tc_padded_compact_rows_dev_);
  if (tc_int_workspace_dev_ != nullptr) cudaFree(tc_int_workspace_dev_);
  if (tc_float_workspace_dev_ != nullptr) cudaFree(tc_float_workspace_dev_);
  if (tc_group_int_workspace_dev_ != nullptr) cudaFree(tc_group_int_workspace_dev_);
  if (tc_group_float_workspace_dev_ != nullptr) cudaFree(tc_group_float_workspace_dev_);

  tc_max_rows_ = padded_rows;
  constexpr size_t kCutlassWorkspaceBytes = 32ull * 1024ull * 1024ull;
  cudaError_t e = cudaSuccess;
  e = cudaMalloc(&tc_a_fp8_dev_, static_cast<size_t>(padded_rows) * hidden_ * sizeof(uint8_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_a_fp8_dev_");
  e = cudaMalloc(&tc_b_col_dev_, static_cast<size_t>(gemm1_out_) * hidden_ * sizeof(uint8_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_b_col_dev_");
  e = cudaMalloc(&tc_a_scale_dev_, static_cast<size_t>(padded_rows) * hidden_blocks_ * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_a_scale_dev_");
  size_t tc_b_scale_elems =
      std::max(static_cast<size_t>(gemm1_out_blocks_) * hidden_blocks_,
               static_cast<size_t>(hidden_) * intermediate_blocks_);
  e = cudaMalloc(&tc_b_scale_dev_, tc_b_scale_elems * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_b_scale_dev_");
  e = cudaMalloc(&tc_g1_f32_dev_, static_cast<size_t>(padded_rows) * gemm1_out_ * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_g1_f32_dev_");
  e = cudaMalloc(&tc_c_fp8_dev_, static_cast<size_t>(padded_rows) * intermediate_ * sizeof(uint8_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_c_fp8_dev_");
  e = cudaMalloc(&tc_c_scale_dev_, static_cast<size_t>(padded_rows) * intermediate_blocks_ * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_c_scale_dev_");
  e = cudaMalloc(&tc_d_f32_dev_, static_cast<size_t>(padded_rows) * hidden_ * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_d_f32_dev_");
  e = cudaMalloc(&tc_m_indptr_dev_, 33 * sizeof(int));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_m_indptr_dev_");
  e = cudaMalloc(&tc_padded_compact_rows_dev_, static_cast<size_t>(padded_rows) * sizeof(int));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_padded_compact_rows_dev_");
  if (tc_m_indptr_host_ == nullptr) {
    e = cudaHostAlloc(&tc_m_indptr_host_, 33 * sizeof(int), cudaHostAllocDefault);
    if (e != cudaSuccess) tc_m_indptr_host_ = nullptr;
  }
  e = cudaMalloc(&tc_int_workspace_dev_, kCutlassWorkspaceBytes);
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_int_workspace_dev_");
  e = cudaMalloc(&tc_float_workspace_dev_, kCutlassWorkspaceBytes);
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_float_workspace_dev_");
  e = cudaMalloc(&tc_group_int_workspace_dev_, kCutlassWorkspaceBytes);
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_group_int_workspace_dev_");
  e = cudaMalloc(&tc_group_float_workspace_dev_, kCutlassWorkspaceBytes);
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_group_float_workspace_dev_");
}

void DeviceMxfpGemmModule::RunExpert(const float* a_dev, int64_t t, const float* local_weight_dev,
                                     int local_expert_idx, const uint8_t* gemm1_w_dev,
                                     const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
                                     const float* gemm2_s_dev, float* out_acc_dev,
                                     cudaStream_t stream) const {
  size_t w13_elems = static_cast<size_t>(gemm1_out_) * static_cast<size_t>(hidden_);
  size_t w13s_elems = static_cast<size_t>(gemm1_out_blocks_) * static_cast<size_t>(hidden_blocks_);
  size_t w2_elems = static_cast<size_t>(hidden_) * static_cast<size_t>(intermediate_);
  size_t w2s_elems = static_cast<size_t>(hidden_blocks_) * static_cast<size_t>(intermediate_blocks_);

  const uint8_t* w13_e = gemm1_w_dev + static_cast<size_t>(local_expert_idx) * w13_elems;
  const float* s13_e = gemm1_s_dev + static_cast<size_t>(local_expert_idx) * w13s_elems;
  const uint8_t* w2_e = gemm2_w_dev + static_cast<size_t>(local_expert_idx) * w2_elems;
  const float* s2_e = gemm2_s_dev + static_cast<size_t>(local_expert_idx) * w2s_elems;
  constexpr int kThreads = 128;
  int64_t n_g1 = t * gemm1_out_;
  int64_t n_c = t * intermediate_;
  int64_t n_out = t * hidden_;

  int b1 = static_cast<int>((n_g1 + kThreads - 1) / kThreads);
  int b2 = static_cast<int>((n_c + kThreads - 1) / kThreads);
  int b3 = static_cast<int>((n_out + kThreads - 1) / kThreads);

  gemm1_kernel<<<b1, kThreads, 0, stream>>>(a_dev, t, hidden_, gemm1_out_, block_, hidden_blocks_,
                                            local_expert_idx, local_weight_dev, w13_e, s13_e,
                                            emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_, g1_dev_);
  swiglu_kernel<<<b2, kThreads, 0, stream>>>(g1_dev_, t, intermediate_, local_expert_idx,
                                             local_weight_dev, emulate_fp8_unit_, c_dev_);
  gemm2_acc_kernel<<<b3, kThreads, 0, stream>>>(c_dev_, t, hidden_, intermediate_, block_,
                                                intermediate_blocks_, local_expert_idx,
                                                local_weight_dev, w2_e, s2_e, emulate_fp8_unit_,
                                                emulate_fp16_operands_,
                                                emulate_acc_half_,
                                                out_acc_dev);
}

void DeviceMxfpGemmModule::RunExpertPermuted(const float* a_dev, int64_t /*t*/, int n_rows,
                                             const int* permuted_tok_e,
                                             const float* permuted_w_e, int local_expert_idx,
                                             const uint8_t* gemm1_w_dev, const float* gemm1_s_dev,
                                             const uint8_t* gemm2_w_dev, const float* gemm2_s_dev,
                                             float* out_acc_dev, cudaStream_t stream) const {
  if (n_rows <= 0) return;

  size_t w13_elems = static_cast<size_t>(gemm1_out_) * static_cast<size_t>(hidden_);
  size_t w13s_elems = static_cast<size_t>(gemm1_out_blocks_) * static_cast<size_t>(hidden_blocks_);
  size_t w2_elems = static_cast<size_t>(hidden_) * static_cast<size_t>(intermediate_);
  size_t w2s_elems = static_cast<size_t>(hidden_blocks_) * static_cast<size_t>(intermediate_blocks_);

  const uint8_t* w13_e = gemm1_w_dev + static_cast<size_t>(local_expert_idx) * w13_elems;
  const float* s13_e = gemm1_s_dev + static_cast<size_t>(local_expert_idx) * w13s_elems;
  const uint8_t* w2_e = gemm2_w_dev + static_cast<size_t>(local_expert_idx) * w2_elems;
  const float* s2_e = gemm2_s_dev + static_cast<size_t>(local_expert_idx) * w2s_elems;

  constexpr int kThreads = 128;
  int64_t n_g1 = static_cast<int64_t>(n_rows) * gemm1_out_;
  int64_t n_c = static_cast<int64_t>(n_rows) * intermediate_;
  int64_t n_out = static_cast<int64_t>(n_rows) * hidden_;

  int b1 = static_cast<int>((n_g1 + kThreads - 1) / kThreads);
  int b2 = static_cast<int>((n_c + kThreads - 1) / kThreads);
  int b3 = static_cast<int>((n_out + kThreads - 1) / kThreads);

  gemm1_permuted_kernel<<<b1, kThreads, 0, stream>>>(
      a_dev, hidden_, gemm1_out_, block_, hidden_blocks_, n_rows, permuted_tok_e, w13_e, s13_e,
      emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_, g1_dev_);
  swiglu_permuted_kernel<<<b2, kThreads, 0, stream>>>(g1_dev_, intermediate_, n_rows,
                                                      emulate_fp8_unit_, c_dev_);
  gemm2_scatter_accumulate_kernel<<<b3, kThreads, 0, stream>>>(
      c_dev_, hidden_, intermediate_, block_, intermediate_blocks_, n_rows, permuted_tok_e,
      permuted_w_e, w2_e, s2_e, emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_,
      out_acc_dev);
}

void DeviceMxfpGemmModule::RunExpertGemm2TcFromG1(
    int64_t t, int total_rows, int n_rows, int g1_row_offset, int scratch_row_offset,
    const int* permuted_tok_e, const float* permuted_w_e, int local_expert_idx,
    const uint8_t* gemm2_w_dev,
    const float* gemm2_s_dev, float* out_acc_dev, cudaStream_t stream, bool scatter_to_acc) {
  if (n_rows <= 0) return;
#if FIB_HAS_DIRECT_CUTLASS_SM100
  constexpr int kThreads = 256;
  using FP8 = cutlass::float_e4m3_t;
  int padded_rows = (n_rows + 3) & ~3;
  const bool scale_major_k = true;
  size_t w2_elems = static_cast<size_t>(hidden_) * static_cast<size_t>(intermediate_);
  size_t w2s_elems = static_cast<size_t>(hidden_blocks_) * static_cast<size_t>(intermediate_blocks_);
  const uint8_t* w2_e = gemm2_w_dev + static_cast<size_t>(local_expert_idx) * w2_elems;
  const float* s2_e = gemm2_s_dev + static_cast<size_t>(local_expert_idx) * w2s_elems;

  float* g1_ptr = tc_g1_f32_dev_ + static_cast<int64_t>(g1_row_offset) * gemm1_out_;
  float* d_ptr = tc_d_f32_dev_ + static_cast<int64_t>(scratch_row_offset) * hidden_;
  float* c_scale_ptr = tc_c_scale_dev_;
  int swiglu_rows_per_cta = select_swiglu_rows_per_cta(total_rows, padded_rows);
  dim3 qgrid((padded_rows + swiglu_rows_per_cta - 1) / swiglu_rows_per_cta,
             intermediate_blocks_);
  if (swiglu_rows_per_cta == 4) {
    swiglu_quantize_float_to_fp8_rows_per_cta_kernel<4>
        <<<qgrid, 128, 4 * 128 * sizeof(float), stream>>>(
            g1_ptr, intermediate_, n_rows, padded_rows,
            tc_c_fp8_dev_ + static_cast<int64_t>(scratch_row_offset) * intermediate_, c_scale_ptr,
            scale_major_k);
  } else if (swiglu_rows_per_cta == 2) {
    swiglu_quantize_float_to_fp8_rows_per_cta_kernel<2>
        <<<qgrid, 128, 2 * 128 * sizeof(float), stream>>>(
            g1_ptr, intermediate_, n_rows, padded_rows,
            tc_c_fp8_dev_ + static_cast<int64_t>(scratch_row_offset) * intermediate_, c_scale_ptr,
            scale_major_k);
  } else {
    swiglu_quantize_float_to_fp8_kernel<<<qgrid, 128, 128 * sizeof(float), stream>>>(
        g1_ptr, intermediate_, n_rows, padded_rows,
        tc_c_fp8_dev_ + static_cast<int64_t>(scratch_row_offset) * intermediate_, c_scale_ptr,
        scale_major_k);
  }
  float* w2_scale_tc = const_cast<float*>(s2_e);

  TcGemmDispatch g2_dispatch = select_trtllm_like_gemm2_dispatch(total_rows, padded_rows);
  int g2_tile_m = select_tile_m_from_env("FIB_MOE_G2_TILE_M", g2_dispatch.tile_m);
  auto run_gemm2_1sm = [&]() {
    if (g2_tile_m == 64) {
      return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 64, 1, cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull,
          reinterpret_cast<FP8*>(tc_c_fp8_dev_ + static_cast<int64_t>(scratch_row_offset) *
                                                    intermediate_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(w2_e)),
          c_scale_ptr,
          w2_scale_tc, d_ptr, padded_rows, hidden_, intermediate_, stream);
    }
    return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 128, 1, cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull,
        reinterpret_cast<FP8*>(tc_c_fp8_dev_ + static_cast<int64_t>(scratch_row_offset) *
                                                  intermediate_),
        const_cast<FP8*>(reinterpret_cast<const FP8*>(w2_e)),
        c_scale_ptr,
        w2_scale_tc, d_ptr, padded_rows, hidden_, intermediate_, stream);
  };
  auto run_gemm2_2sm = [&]() {
    return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 256, 2, cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull,
        reinterpret_cast<FP8*>(tc_c_fp8_dev_ + static_cast<int64_t>(scratch_row_offset) *
                                                  intermediate_),
        const_cast<FP8*>(reinterpret_cast<const FP8*>(w2_e)),
        c_scale_ptr,
        w2_scale_tc, d_ptr, padded_rows, hidden_, intermediate_, stream);
  };
  const char* env_g2_2sm_min_rows = std::getenv("FIB_MOE_G2_2SM_MIN_ROWS");
  const char* env_g2_mma_sm = std::getenv("FIB_MOE_G2_MMA_SM");
  int g2_mma_sm = select_mma_sm_from_env_value(env_g2_mma_sm, g2_dispatch.mma_sms);
  if (env_g2_mma_sm == nullptr && env_g2_2sm_min_rows != nullptr) {
    g2_mma_sm = (padded_rows >= std::atoi(env_g2_2sm_min_rows)) ? 2 : 1;
  }
  cudaError_t st2 = (g2_mma_sm == 2) ? run_gemm2_2sm() : run_gemm2_1sm();
  if (st2 != cudaSuccess) return;
  if (!scatter_to_acc) return;

  int64_t scatter_elems = static_cast<int64_t>(n_rows) * hidden_;
  scatter_float_weighted_kernel<<<static_cast<int>((scatter_elems + kThreads - 1) / kThreads),
                                  kThreads, 0, stream>>>(
      d_ptr, hidden_, n_rows, permuted_tok_e, permuted_w_e, out_acc_dev);
#else
  (void)t;
  (void)total_rows;
  (void)n_rows;
  (void)g1_row_offset;
  (void)scratch_row_offset;
  (void)permuted_tok_e;
  (void)permuted_w_e;
  (void)local_expert_idx;
  (void)gemm2_w_dev;
  (void)gemm2_s_dev;
  (void)out_acc_dev;
  (void)stream;
  (void)scatter_to_acc;
#endif
}

void DeviceMxfpGemmModule::RunExpertPermutedTcToScratch(
    const uint8_t* hidden_fp8_dev, const float* hidden_scale_dev, int64_t t, int total_rows,
    int row_offset, int n_rows, const int* permuted_tok_e, int local_expert_idx,
    const uint8_t* gemm1_w_dev, const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
    const float* gemm2_s_dev, cudaStream_t stream, bool run_gemm2) {
  if (n_rows <= 0) return;
#if FIB_HAS_DIRECT_CUTLASS_SM100
  const bool nvtx = mxfp_nvtx_enabled();
  ScopedMxfpNvtxRange nvtx_total(nvtx, "mxfp_expert_tc_to_scratch");
  EnsureTcWorkspace(total_rows);
  constexpr int kThreads = 256;
  using FP8 = cutlass::float_e4m3_t;
  int padded_rows = (n_rows + 3) & ~3;
  const int c_row_blocks = (padded_rows + 127) / 128;

  uint8_t* a_ptr = tc_a_fp8_dev_ + static_cast<int64_t>(row_offset) * hidden_;
  float* a_scale_ptr = tc_a_scale_dev_ + static_cast<int64_t>(row_offset) * hidden_blocks_;
  float* g1_ptr = tc_g1_f32_dev_ + static_cast<int64_t>(row_offset) * gemm1_out_;
  float* d_ptr = tc_d_f32_dev_ + static_cast<int64_t>(row_offset) * hidden_;
  float* c_scale_ptr =
      tc_c_scale_dev_ + static_cast<int64_t>(run_gemm2 ? 0 : row_offset) * intermediate_blocks_;

  size_t w13_elems = static_cast<size_t>(gemm1_out_) * static_cast<size_t>(hidden_);
  size_t w13s_elems = static_cast<size_t>(gemm1_out_blocks_) * static_cast<size_t>(hidden_blocks_);
  size_t w2_elems = static_cast<size_t>(hidden_) * static_cast<size_t>(intermediate_);
  size_t w2s_elems = static_cast<size_t>(hidden_blocks_) * static_cast<size_t>(intermediate_blocks_);
  const uint8_t* w13_e = gemm1_w_dev + static_cast<size_t>(local_expert_idx) * w13_elems;
  const float* s13_e = gemm1_s_dev + static_cast<size_t>(local_expert_idx) * w13s_elems;
  const uint8_t* w2_e = gemm2_w_dev + static_cast<size_t>(local_expert_idx) * w2_elems;
  const float* s2_e = gemm2_s_dev + static_cast<size_t>(local_expert_idx) * w2s_elems;

  int hidden_chunks16 = hidden_ / 16;
  bool vec16_gather = (hidden_ % 16 == 0) && use_vec16_hidden_gather(padded_rows);
  int64_t a_elems =
      static_cast<int64_t>(padded_rows) * (vec16_gather ? hidden_chunks16 : hidden_);
  int64_t a_scale_elems = static_cast<int64_t>(padded_rows) * hidden_blocks_;
  {
    ScopedMxfpNvtxRange range(nvtx, "mxfp_gather_hidden_fp8_scale");
    if (vec16_gather) {
      gather_hidden_fp8_vec16_and_scale_rows_mn_major_kernel<<<
          static_cast<int>((std::max(a_elems, a_scale_elems) + kThreads - 1) / kThreads),
          kThreads, 0, stream>>>(hidden_fp8_dev, hidden_scale_dev, permuted_tok_e, t, n_rows,
                                 padded_rows, hidden_chunks16, hidden_, hidden_blocks_, a_ptr,
                                 a_scale_ptr);
    } else {
      gather_hidden_fp8_and_scale_rows_mn_major_kernel<<<
          static_cast<int>((std::max(a_elems, a_scale_elems) + kThreads - 1) / kThreads),
          kThreads, 0, stream>>>(hidden_fp8_dev, hidden_scale_dev, permuted_tok_e, t, n_rows,
                                 padded_rows, hidden_, hidden_blocks_, a_ptr, a_scale_ptr);
    }
  }

  float* w13_scale_tc = const_cast<float*>(s13_e);

  TcGemmDispatch g1_dispatch = select_trtllm_like_gemm1_dispatch(total_rows, padded_rows);
  int g1_tile_m = select_tile_m_from_env("FIB_MOE_G1_TILE_M", g1_dispatch.tile_m);
  auto run_gemm1_1sm = [&]() {
    if (g1_tile_m == 64) {
      return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 64, 1,
                                                         cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(a_ptr),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(w13_e)), a_scale_ptr, w13_scale_tc,
          g1_ptr, padded_rows, gemm1_out_, hidden_, stream);
    }
    return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 128, 1,
                                                       cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(a_ptr),
        const_cast<FP8*>(reinterpret_cast<const FP8*>(w13_e)), a_scale_ptr, w13_scale_tc,
        g1_ptr, padded_rows, gemm1_out_, hidden_, stream);
  };
  auto run_gemm1_2sm = [&]() {
    return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 256, 2,
                                                       cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(a_ptr),
        const_cast<FP8*>(reinterpret_cast<const FP8*>(w13_e)), a_scale_ptr, w13_scale_tc,
        g1_ptr, padded_rows, gemm1_out_, hidden_, stream);
  };
  int g1_mma_sm = select_mma_sm_from_env("FIB_MOE_G1_MMA_SM", g1_dispatch.mma_sms);
  cudaError_t st1 = cudaSuccess;
  {
    ScopedMxfpNvtxRange range(nvtx, "mxfp_cutlass_gemm1");
    st1 = (g1_mma_sm == 2) ? run_gemm1_2sm() : run_gemm1_1sm();
  }
  if (st1 != cudaSuccess) return;

  {
    ScopedMxfpNvtxRange range(nvtx, "mxfp_swiglu_quant");
    if (tc_fp16_middle_) {
      dim3 qgrid(padded_rows, intermediate_blocks_);
      uint16_t* g1_f16_ptr = reinterpret_cast<uint16_t*>(tc_float_workspace_dev_);
      float_to_f16_kernel<<<
          static_cast<int>(((static_cast<int64_t>(padded_rows) * gemm1_out_) + kThreads - 1) /
                           kThreads),
          kThreads, 0, stream>>>(g1_ptr, static_cast<int64_t>(padded_rows) * gemm1_out_,
                                 g1_f16_ptr);
      swiglu_quantize_f16_to_fp8_kernel<<<qgrid, 128, 128 * sizeof(float), stream>>>(
          g1_f16_ptr, intermediate_, n_rows, padded_rows,
          tc_c_fp8_dev_ + static_cast<int64_t>(row_offset) * intermediate_, c_scale_ptr, true);
    } else {
      int swiglu_rows_per_cta = select_swiglu_rows_per_cta(n_rows, padded_rows);
      dim3 qgrid((padded_rows + swiglu_rows_per_cta - 1) / swiglu_rows_per_cta,
                 intermediate_blocks_);
      if (swiglu_rows_per_cta == 4) {
        swiglu_quantize_float_to_fp8_rows_per_cta_kernel<4>
            <<<qgrid, 128, 4 * 128 * sizeof(float), stream>>>(
                g1_ptr, intermediate_, n_rows, padded_rows,
                tc_c_fp8_dev_ + static_cast<int64_t>(row_offset) * intermediate_, c_scale_ptr, true);
      } else if (swiglu_rows_per_cta == 2) {
        swiglu_quantize_float_to_fp8_rows_per_cta_kernel<2>
            <<<qgrid, 128, 2 * 128 * sizeof(float), stream>>>(
                g1_ptr, intermediate_, n_rows, padded_rows,
                tc_c_fp8_dev_ + static_cast<int64_t>(row_offset) * intermediate_, c_scale_ptr, true);
      } else {
        swiglu_quantize_float_to_fp8_kernel<<<qgrid, 128, 128 * sizeof(float), stream>>>(
            g1_ptr, intermediate_, n_rows, padded_rows,
            tc_c_fp8_dev_ + static_cast<int64_t>(row_offset) * intermediate_, c_scale_ptr, true);
      }
    }
  }
  if (!run_gemm2) return;
  float* w2_scale_tc = const_cast<float*>(s2_e);

  TcGemmDispatch g2_dispatch = select_trtllm_like_gemm2_dispatch(total_rows, padded_rows);
  int g2_tile_m = select_tile_m_from_env("FIB_MOE_G2_TILE_M", g2_dispatch.tile_m);
  auto run_gemm2_1sm = [&]() {
    if (g2_tile_m == 64) {
      return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 64, 1,
                                                         cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull,
          tc_float_workspace_dev_, 32ull * 1024ull * 1024ull,
          reinterpret_cast<FP8*>(tc_c_fp8_dev_ + static_cast<int64_t>(row_offset) * intermediate_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(w2_e)),
          c_scale_ptr,
          w2_scale_tc,
          d_ptr, padded_rows, hidden_, intermediate_, stream);
    }
    return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 128, 1,
                                                       cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull,
        tc_float_workspace_dev_, 32ull * 1024ull * 1024ull,
        reinterpret_cast<FP8*>(tc_c_fp8_dev_ + static_cast<int64_t>(row_offset) * intermediate_),
        const_cast<FP8*>(reinterpret_cast<const FP8*>(w2_e)),
        c_scale_ptr,
        w2_scale_tc,
        d_ptr, padded_rows, hidden_, intermediate_, stream);
  };
  auto run_gemm2_2sm = [&]() {
    return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 256, 2,
                                                       cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull,
        tc_float_workspace_dev_, 32ull * 1024ull * 1024ull,
        reinterpret_cast<FP8*>(tc_c_fp8_dev_ + static_cast<int64_t>(row_offset) * intermediate_),
        const_cast<FP8*>(reinterpret_cast<const FP8*>(w2_e)),
        c_scale_ptr,
        w2_scale_tc,
        d_ptr, padded_rows, hidden_, intermediate_, stream);
  };
  const char* env_g2_2sm_min_rows = std::getenv("FIB_MOE_G2_2SM_MIN_ROWS");
  const char* env_g2_mma_sm = std::getenv("FIB_MOE_G2_MMA_SM");
  int g2_mma_sm = select_mma_sm_from_env_value(env_g2_mma_sm, g2_dispatch.mma_sms);
  if (env_g2_mma_sm == nullptr && env_g2_2sm_min_rows != nullptr) {
    g2_mma_sm = (padded_rows >= std::atoi(env_g2_2sm_min_rows)) ? 2 : 1;
  }
  {
    ScopedMxfpNvtxRange range(nvtx, "mxfp_cutlass_gemm2");
    (void)((g2_mma_sm == 2) ? run_gemm2_2sm() : run_gemm2_1sm());
  }
#else
  (void)hidden_fp8_dev;
  (void)hidden_scale_dev;
  (void)t;
  (void)total_rows;
  (void)row_offset;
  (void)n_rows;
  (void)permuted_tok_e;
  (void)local_expert_idx;
  (void)gemm1_w_dev;
  (void)gemm1_s_dev;
  (void)gemm2_w_dev;
  (void)gemm2_s_dev;
  (void)stream;
  (void)run_gemm2;
#endif
}

void DeviceMxfpGemmModule::ScatterTcScratch(int total_rows, const int* expert_offsets_dev,
                                            const int* permuted_tok_dev,
                                            const int* permuted_expert_dev,
                                            const float* permuted_w_dev, float* out_acc_dev,
                                            cudaStream_t stream) {
  if (total_rows <= 0) return;
  (void)expert_offsets_dev;
  constexpr int kThreads = 256;
  dim3 grid(total_rows, (hidden_ + kThreads - 1) / kThreads);
  scatter_all_float_weighted_by_expert_row_kernel<<<grid, kThreads, 0, stream>>>(
      tc_d_f32_dev_, hidden_, total_rows, permuted_tok_dev, permuted_expert_dev,
      permuted_w_dev, out_acc_dev);
}

void DeviceMxfpGemmModule::WriteTcScratchToBf16Output(
    int64_t t, const int* routed_positions_dev, const int* routed_local_experts_dev,
    const float* routed_weights_dev, uint16_t* output_dev, cudaStream_t stream) {
  if (t <= 0) return;
  const bool nvtx = mxfp_nvtx_enabled();
  ScopedMxfpNvtxRange range(nvtx, "mxfp_direct_output_gather");
  int kThreads = select_direct_output_threads(t);
  int vec = select_direct_output_vec(t);
  dim3 grid(static_cast<unsigned int>(t), (hidden_ + kThreads * vec - 1) / (kThreads * vec));
  if (vec == 8) {
    if (hidden_ % (kThreads * 8) == 0) {
      gather_scratch_topk_to_bf16_vec_kernel<8, true><<<grid, kThreads, 0, stream>>>(
          tc_d_f32_dev_, hidden_, t, routed_positions_dev, routed_local_experts_dev,
          routed_weights_dev, output_dev);
    } else {
      gather_scratch_topk_to_bf16_vec_kernel<8><<<grid, kThreads, 0, stream>>>(
          tc_d_f32_dev_, hidden_, t, routed_positions_dev, routed_local_experts_dev,
          routed_weights_dev, output_dev);
    }
  } else if (vec == 4) {
    if (hidden_ % (kThreads * 4) == 0) {
      gather_scratch_topk_to_bf16_vec_kernel<4, true><<<grid, kThreads, 0, stream>>>(
          tc_d_f32_dev_, hidden_, t, routed_positions_dev, routed_local_experts_dev,
          routed_weights_dev, output_dev);
    } else {
      gather_scratch_topk_to_bf16_vec_kernel<4><<<grid, kThreads, 0, stream>>>(
          tc_d_f32_dev_, hidden_, t, routed_positions_dev, routed_local_experts_dev,
          routed_weights_dev, output_dev);
    }
  } else if (vec == 2) {
    if (hidden_ % (kThreads * 2) == 0) {
      gather_scratch_topk_to_bf16_vec_kernel<2, true><<<grid, kThreads, 0, stream>>>(
          tc_d_f32_dev_, hidden_, t, routed_positions_dev, routed_local_experts_dev,
          routed_weights_dev, output_dev);
    } else {
      gather_scratch_topk_to_bf16_vec_kernel<2><<<grid, kThreads, 0, stream>>>(
          tc_d_f32_dev_, hidden_, t, routed_positions_dev, routed_local_experts_dev,
          routed_weights_dev, output_dev);
    }
  } else {
    dim3 scalar_grid(static_cast<unsigned int>(t), (hidden_ + kThreads - 1) / kThreads);
    gather_scratch_topk_to_bf16_kernel<<<scalar_grid, kThreads, 0, stream>>>(
        tc_d_f32_dev_, hidden_, t, routed_positions_dev, routed_local_experts_dev,
        routed_weights_dev, output_dev);
  }
}

void DeviceMxfpGemmModule::WriteTcPaddedScratchToBf16Output(
    int64_t t, const int* expert_offsets_dev, const int* padded_offsets_dev,
    const int* routed_positions_dev, const int* routed_local_experts_dev,
    const float* routed_weights_dev, uint16_t* output_dev, cudaStream_t stream) {
  if (t <= 0) return;
  const bool nvtx = mxfp_nvtx_enabled();
  ScopedMxfpNvtxRange range(nvtx, "mxfp_padded_direct_output_gather");
  int kThreads = select_direct_output_threads(t);
  int vec = select_direct_output_vec(t);
  dim3 grid(static_cast<unsigned int>(t), (hidden_ + kThreads * vec - 1) / (kThreads * vec));
  if (vec == 8) {
    if (hidden_ % (kThreads * 8) == 0) {
      gather_padded_scratch_topk_to_bf16_vec_kernel<8, true><<<grid, kThreads, 0, stream>>>(
          tc_d_f32_dev_, hidden_, t, expert_offsets_dev, padded_offsets_dev, routed_positions_dev,
          routed_local_experts_dev, routed_weights_dev, output_dev);
    } else {
      gather_padded_scratch_topk_to_bf16_vec_kernel<8><<<grid, kThreads, 0, stream>>>(
          tc_d_f32_dev_, hidden_, t, expert_offsets_dev, padded_offsets_dev, routed_positions_dev,
          routed_local_experts_dev, routed_weights_dev, output_dev);
    }
  } else if (vec == 4) {
    if (hidden_ % (kThreads * 4) == 0) {
      gather_padded_scratch_topk_to_bf16_vec_kernel<4, true><<<grid, kThreads, 0, stream>>>(
          tc_d_f32_dev_, hidden_, t, expert_offsets_dev, padded_offsets_dev, routed_positions_dev,
          routed_local_experts_dev, routed_weights_dev, output_dev);
    } else {
      gather_padded_scratch_topk_to_bf16_vec_kernel<4><<<grid, kThreads, 0, stream>>>(
          tc_d_f32_dev_, hidden_, t, expert_offsets_dev, padded_offsets_dev, routed_positions_dev,
          routed_local_experts_dev, routed_weights_dev, output_dev);
    }
  } else if (vec == 2) {
    if (hidden_ % (kThreads * 2) == 0) {
      gather_padded_scratch_topk_to_bf16_vec_kernel<2, true><<<grid, kThreads, 0, stream>>>(
          tc_d_f32_dev_, hidden_, t, expert_offsets_dev, padded_offsets_dev, routed_positions_dev,
          routed_local_experts_dev, routed_weights_dev, output_dev);
    } else {
      gather_padded_scratch_topk_to_bf16_vec_kernel<2><<<grid, kThreads, 0, stream>>>(
          tc_d_f32_dev_, hidden_, t, expert_offsets_dev, padded_offsets_dev, routed_positions_dev,
          routed_local_experts_dev, routed_weights_dev, output_dev);
    }
  } else {
    dim3 scalar_grid(static_cast<unsigned int>(t), (hidden_ + kThreads - 1) / kThreads);
    gather_padded_scratch_topk_to_bf16_kernel<<<scalar_grid, kThreads, 0, stream>>>(
        tc_d_f32_dev_, hidden_, t, expert_offsets_dev, padded_offsets_dev, routed_positions_dev,
        routed_local_experts_dev, routed_weights_dev, output_dev);
  }
}

bool DeviceMxfpGemmModule::RunAllExpertsDenseTc(
    const uint8_t* hidden_fp8_dev, const float* hidden_scale_dev, int64_t t,
    const int* routed_local_experts_dev, const float* routed_weights_dev,
    const uint8_t* gemm1_w_dev, const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
    const float* gemm2_s_dev, uint16_t* output_dev, cudaStream_t stream) {
  if (t <= 0) return true;
#if FIB_HAS_DIRECT_CUTLASS_SM100
  if (!SupportsTcPath()) return false;
  if (hidden_ % 16 != 0) return false;

  const int padded_t = (static_cast<int>(t) + 3) & ~3;
  const int total_rows = 32 * padded_t;
  EnsureTcWorkspace(total_rows);

  std::array<int, 33> offsets_fallback{};
  int* offsets = tc_m_indptr_host_ == nullptr ? offsets_fallback.data() : tc_m_indptr_host_;
  for (int le = 0; le <= 32; ++le) offsets[le] = le * padded_t;
  cudaMemcpyAsync(tc_m_indptr_dev_, offsets, 33 * sizeof(int), cudaMemcpyHostToDevice, stream);

  constexpr int kThreads = 256;
  int hidden_chunks16 = hidden_ / 16;
  int64_t a_elems = static_cast<int64_t>(total_rows) * hidden_chunks16;
  int64_t a_scale_elems = static_cast<int64_t>(total_rows) * hidden_blocks_;
  duplicate_all_expert_hidden_fp8_and_scale_kernel<<<
      static_cast<int>((std::max(a_elems, a_scale_elems) + kThreads - 1) / kThreads), kThreads, 0,
      stream>>>(hidden_fp8_dev, hidden_scale_dev, t, padded_t, hidden_chunks16, hidden_,
                hidden_blocks_, tc_a_fp8_dev_, tc_a_scale_dev_);

  using FP8 = cutlass::float_e4m3_t;
  cudaError_t st1 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
      1, true, 64, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
      tc_group_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_group_float_workspace_dev_,
      32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
      const_cast<FP8*>(reinterpret_cast<const FP8*>(gemm1_w_dev)), tc_a_scale_dev_,
      const_cast<float*>(gemm1_s_dev), tc_g1_f32_dev_, tc_m_indptr_dev_, total_rows,
      gemm1_out_, hidden_, 32, stream, nullptr);
  if (st1 != cudaSuccess) return false;

  dim3 qgrid(total_rows, intermediate_blocks_);
  grouped_swiglu_quantize_to_fp8_kernel<float, false>
      <<<qgrid, 128, 4 * sizeof(float), stream>>>(
          tc_g1_f32_dev_, intermediate_, total_rows, tc_m_indptr_dev_, tc_m_indptr_dev_,
          tc_c_fp8_dev_, tc_c_scale_dev_);

  cudaError_t st2 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
      1, true, 64, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
      tc_group_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_group_float_workspace_dev_,
      32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_),
      const_cast<FP8*>(reinterpret_cast<const FP8*>(gemm2_w_dev)), tc_c_scale_dev_,
      const_cast<float*>(gemm2_s_dev), tc_d_f32_dev_, tc_m_indptr_dev_, total_rows, hidden_,
      intermediate_, 32, stream, nullptr);
  if (st2 != cudaSuccess) return false;

  int threads = select_direct_output_threads(t);
  int vec = select_direct_output_vec(t);
  dim3 grid(static_cast<unsigned int>(t), (hidden_ + threads * vec - 1) / (threads * vec));
  if (vec == 8) {
    gather_all_expert_scratch_topk_to_bf16_vec_kernel<8><<<grid, threads, 0, stream>>>(
        tc_d_f32_dev_, hidden_, t, padded_t, tc_m_indptr_dev_, routed_local_experts_dev,
        routed_weights_dev, output_dev);
  } else if (vec == 4) {
    gather_all_expert_scratch_topk_to_bf16_vec_kernel<4><<<grid, threads, 0, stream>>>(
        tc_d_f32_dev_, hidden_, t, padded_t, tc_m_indptr_dev_, routed_local_experts_dev,
        routed_weights_dev, output_dev);
  } else if (vec == 2) {
    gather_all_expert_scratch_topk_to_bf16_vec_kernel<2><<<grid, threads, 0, stream>>>(
        tc_d_f32_dev_, hidden_, t, padded_t, tc_m_indptr_dev_, routed_local_experts_dev,
        routed_weights_dev, output_dev);
  } else {
    dim3 scalar_grid(static_cast<unsigned int>(t), (hidden_ + threads - 1) / threads);
    gather_all_expert_scratch_topk_to_bf16_vec_kernel<1><<<scalar_grid, threads, 0, stream>>>(
        tc_d_f32_dev_, hidden_, t, padded_t, tc_m_indptr_dev_, routed_local_experts_dev,
        routed_weights_dev, output_dev);
  }
  return cudaGetLastError() == cudaSuccess;
#else
  (void)hidden_fp8_dev;
  (void)hidden_scale_dev;
  (void)t;
  (void)routed_local_experts_dev;
  (void)routed_weights_dev;
  (void)gemm1_w_dev;
  (void)gemm1_s_dev;
  (void)gemm2_w_dev;
  (void)gemm2_s_dev;
  (void)output_dev;
  (void)stream;
  return false;
#endif
}

bool DeviceMxfpGemmModule::RunGroupedGemm1ThenExpertGemm2Tc(
    const uint8_t* hidden_fp8_dev, const float* hidden_scale_dev, int64_t t, int total_rows,
    const int* expert_offsets_dev, const int* expert_counts_host,
    const int* expert_offsets_host, const int* permuted_tok_dev,
    const float* permuted_w_dev, const uint8_t* gemm1_w_dev,
    const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
    const float* gemm2_s_dev, float* out_acc_dev, cudaStream_t stream, bool direct_output,
    const int* routed_positions_dev, const int* routed_local_experts_dev,
    const float* routed_weights_dev, uint16_t* output_dev) {
  if (total_rows <= 0) return true;
  const char* env_enable = std::getenv("FIB_MOE_TC_ENABLE_GROUPED_G1");
  const char* env_disable = std::getenv("FIB_MOE_TC_DISABLE_GROUPED_G1");
  if (env_disable != nullptr && env_disable[0] == '1') return false;
  if (env_enable != nullptr && env_enable[0] != '1') return false;
  if (!SupportsTcPath()) return false;
  if (env_enable == nullptr) {
    int auto_max_rows = std::max(
        0, env_int_or_default("FIB_MOE_TC_GROUPED_G1_AUTO_MAX_ROWS", 32768));
    if (auto_max_rows == 0 || total_rows > auto_max_rows) return false;
  }
  int grouped_g1_min_rows = std::max(0, env_int_or_default("FIB_MOE_TC_GROUPED_G1_MIN_ROWS", 0));
  std::array<int, 32> active_expert_ids{};
  std::array<int, 33> compact_offsets{};
  int active_groups = 0;
  for (int le = 0; le < 32; ++le) {
    if (expert_counts_host[le] <= 0) continue;
    if (expert_counts_host[le] < grouped_g1_min_rows) return false;
    active_expert_ids[active_groups] = le;
    compact_offsets[active_groups] = expert_offsets_host[le];
    ++active_groups;
  }
  if (active_groups == 0) return true;
  compact_offsets[active_groups] = total_rows;
#if FIB_HAS_DIRECT_CUTLASS_SM100
  EnsureTcWorkspace(total_rows + 4 * 32);
  void* grouped_arg_workspace = tc_group_int_workspace_dev_;
  void* grouped_gemm_workspace = tc_group_float_workspace_dev_;
  constexpr int kThreads = 256;
  const char* env_device_group_args = std::getenv("FIB_MOE_TC_DEVICE_GROUP_ARGS");
  const bool device_group_args =
      env_device_group_args == nullptr || env_device_group_args[0] != '0';
  int g1_num_groups = device_group_args ? 32 : active_groups;
  const int* g1_offsets_host = device_group_args ? nullptr : compact_offsets.data();
  const int* g1_group_ids_host = device_group_args ? nullptr : active_expert_ids.data();
  int hidden_chunks16 = hidden_ / 16;
  bool vec16_gather = (hidden_ % 16 == 0) && use_vec16_hidden_gather(total_rows);
  int64_t a_elems =
      static_cast<int64_t>(total_rows) * (vec16_gather ? hidden_chunks16 : hidden_);
  int64_t a_scale_elems = static_cast<int64_t>(total_rows) * hidden_blocks_;
  if (vec16_gather) {
    gather_hidden_fp8_vec16_and_scale_rows_mn_major_kernel<<<
        static_cast<int>((std::max(a_elems, a_scale_elems) + kThreads - 1) / kThreads),
        kThreads, 0, stream>>>(hidden_fp8_dev, hidden_scale_dev, permuted_tok_dev, t, total_rows,
                               total_rows, hidden_chunks16, hidden_, hidden_blocks_,
                               tc_a_fp8_dev_, tc_a_scale_dev_);
  } else {
    gather_hidden_fp8_and_scale_rows_mn_major_kernel<<<
        static_cast<int>((std::max(a_elems, a_scale_elems) + kThreads - 1) / kThreads),
        kThreads, 0, stream>>>(hidden_fp8_dev, hidden_scale_dev, permuted_tok_dev, t, total_rows,
                               total_rows, hidden_, hidden_blocks_, tc_a_fp8_dev_,
                               tc_a_scale_dev_);
  }

  using FP8 = cutlass::float_e4m3_t;
  int g1_mma_sm = 1;
  int grouped_g1_mma_sm_override = env_int_or_default("FIB_MOE_TC_GROUPED_G1_MMA_SM", 0);
  if (grouped_g1_mma_sm_override == 1 || grouped_g1_mma_sm_override == 2) {
    g1_mma_sm = grouped_g1_mma_sm_override;
  }
  int grouped_g1_tile_m = select_tile_m_from_env(
      "FIB_MOE_GROUPED_G1_TILE_M", g1_mma_sm == 2 ? 256 : 64);
  const char* env_group_flashinfer = std::getenv("FIB_MOE_TC_GROUPED_G1_FLASHINFER");
  const bool group_flashinfer =
      (env_group_flashinfer != nullptr && env_group_flashinfer[0] == '1' && active_groups == 32);
  const char* env_group_scale_mn = std::getenv("FIB_MOE_TC_GROUPED_G1_SCALE_MN");
  const bool group_scale_mn = (env_group_scale_mn != nullptr && env_group_scale_mn[0] == '1');
  const char* env_grouped_g1_tma_epilogue =
      std::getenv("FIB_MOE_TC_GROUPED_G1_TMA_EPILOGUE");
  const bool grouped_g1_tma_epilogue =
      env_grouped_g1_tma_epilogue != nullptr && env_grouped_g1_tma_epilogue[0] == '1';
  const char* env_grouped_g2_after_g1 = std::getenv("FIB_MOE_TC_GROUPED_G1_GROUPED_G2");
  int grouped_g2_after_g1_max_rows =
      std::max(0, env_int_or_default("FIB_MOE_TC_GROUPED_G1_GROUPED_G2_MAX_ROWS", 32768));
  const bool grouped_g2_after_g1_candidate =
      direct_output && grouped_g2_after_g1_max_rows > 0 && total_rows <= grouped_g2_after_g1_max_rows &&
      (env_grouped_g2_after_g1 == nullptr || env_grouped_g2_after_g1[0] != '0');
  const char* env_grouped_g1_f16_output = std::getenv("FIB_MOE_TC_GROUPED_G1_F16_OUTPUT");
  const bool grouped_g1_f16_output = env_grouped_g1_f16_output != nullptr &&
                                     env_grouped_g1_f16_output[0] == '1' &&
                                     !group_scale_mn && grouped_g2_after_g1_candidate;
  std::array<int, 33> padded_offsets_fallback{};
  int* padded_offsets =
      tc_m_indptr_host_ == nullptr ? padded_offsets_fallback.data() : tc_m_indptr_host_;
  padded_offsets[0] = 0;
  for (int le = 0; le < 32; ++le) {
    padded_offsets[le + 1] = padded_offsets[le] + ((expert_counts_host[le] + 3) & ~3);
  }
  int padded_total_rows = padded_offsets[32];
  const uint8_t* g1_w_dev = gemm1_w_dev;
  const float* g1_s_dev = gemm1_s_dev;
  cudaError_t st1 = cudaSuccess;
#if FIB_HAS_FLASHINFER_GROUP_GEMM_FP8_SM100
  if (group_flashinfer) {
    if (group_scale_mn) {
      st1 = (g1_mma_sm == 2)
                ? launch_flashinfer_grouped_blockscaled_gemm1_sm100<false, 2, float>(
                      grouped_arg_workspace, 32ull * 1024ull * 1024ull, grouped_gemm_workspace,
                      32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
                      const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
                      const_cast<float*>(g1_s_dev), tc_g1_f32_dev_,
                      const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_, 32,
                      stream)
                : launch_flashinfer_grouped_blockscaled_gemm1_sm100<false, 1, float>(
                      grouped_arg_workspace, 32ull * 1024ull * 1024ull, grouped_gemm_workspace,
                      32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
                      const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
                      const_cast<float*>(g1_s_dev), tc_g1_f32_dev_,
                      const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_, 32,
                      stream);
    } else {
      st1 = (g1_mma_sm == 2)
                ? launch_flashinfer_grouped_blockscaled_gemm1_sm100<true, 2, float>(
                      grouped_arg_workspace, 32ull * 1024ull * 1024ull, grouped_gemm_workspace,
                      32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
                      const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
                      const_cast<float*>(g1_s_dev), tc_g1_f32_dev_,
                      const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_, 32,
                      stream)
                : launch_flashinfer_grouped_blockscaled_gemm1_sm100<true, 1, float>(
                      grouped_arg_workspace, 32ull * 1024ull * 1024ull, grouped_gemm_workspace,
                      32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
                      const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
                      const_cast<float*>(g1_s_dev), tc_g1_f32_dev_,
                      const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_, 32,
                      stream);
    }
  } else
#endif
  if (group_scale_mn) {
    if (g1_mma_sm == 2) {
      st1 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_select_epilogue<
          1, false, 256, 2, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
          grouped_g1_tma_epilogue, grouped_arg_workspace, 32ull * 1024ull * 1024ull,
          grouped_gemm_workspace,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
          const_cast<float*>(g1_s_dev), tc_g1_f32_dev_,
          const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_,
          g1_num_groups, stream, g1_offsets_host, g1_group_ids_host);
    } else if (grouped_g1_tile_m == 64) {
      st1 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_select_epilogue<
          1, false, 64, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
          grouped_g1_tma_epilogue, grouped_arg_workspace, 32ull * 1024ull * 1024ull,
          grouped_gemm_workspace,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
          const_cast<float*>(g1_s_dev), tc_g1_f32_dev_,
          const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_,
          g1_num_groups, stream, g1_offsets_host, g1_group_ids_host);
    } else {
      st1 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_select_epilogue<
          1, false, 128, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
          grouped_g1_tma_epilogue, grouped_arg_workspace, 32ull * 1024ull * 1024ull,
          grouped_gemm_workspace,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
          const_cast<float*>(g1_s_dev), tc_g1_f32_dev_,
          const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_,
          g1_num_groups, stream, g1_offsets_host, g1_group_ids_host);
    }
  } else {
    if (grouped_g1_f16_output && g1_mma_sm == 2) {
      st1 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_select_epilogue<
          1, true, 256, 2, cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t>(
          grouped_g1_tma_epilogue, grouped_arg_workspace, 32ull * 1024ull * 1024ull,
          grouped_gemm_workspace, 32ull * 1024ull * 1024ull,
          reinterpret_cast<FP8*>(tc_a_fp8_dev_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
          const_cast<float*>(g1_s_dev), reinterpret_cast<cutlass::half_t*>(tc_g1_f32_dev_),
          const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_,
          g1_num_groups, stream, g1_offsets_host, g1_group_ids_host);
    } else if (grouped_g1_f16_output && grouped_g1_tile_m == 64) {
      st1 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_select_epilogue<
          1, true, 64, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t>(
          grouped_g1_tma_epilogue, grouped_arg_workspace, 32ull * 1024ull * 1024ull,
          grouped_gemm_workspace, 32ull * 1024ull * 1024ull,
          reinterpret_cast<FP8*>(tc_a_fp8_dev_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
          const_cast<float*>(g1_s_dev), reinterpret_cast<cutlass::half_t*>(tc_g1_f32_dev_),
          const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_,
          g1_num_groups, stream, g1_offsets_host, g1_group_ids_host);
    } else if (grouped_g1_f16_output) {
      st1 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_select_epilogue<
          1, true, 128, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t>(
          grouped_g1_tma_epilogue, grouped_arg_workspace, 32ull * 1024ull * 1024ull,
          grouped_gemm_workspace, 32ull * 1024ull * 1024ull,
          reinterpret_cast<FP8*>(tc_a_fp8_dev_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
          const_cast<float*>(g1_s_dev), reinterpret_cast<cutlass::half_t*>(tc_g1_f32_dev_),
          const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_,
          g1_num_groups, stream, g1_offsets_host, g1_group_ids_host);
    } else if (g1_mma_sm == 2) {
      st1 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_select_epilogue<
          1, true, 256, 2, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
          grouped_g1_tma_epilogue, grouped_arg_workspace, 32ull * 1024ull * 1024ull,
          grouped_gemm_workspace,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
          const_cast<float*>(g1_s_dev), tc_g1_f32_dev_,
          const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_,
          g1_num_groups, stream, g1_offsets_host, g1_group_ids_host);
    } else if (grouped_g1_tile_m == 64) {
      st1 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_select_epilogue<
          1, true, 64, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
          grouped_g1_tma_epilogue, grouped_arg_workspace, 32ull * 1024ull * 1024ull,
          grouped_gemm_workspace,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
          const_cast<float*>(g1_s_dev), tc_g1_f32_dev_,
          const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_,
          g1_num_groups, stream, g1_offsets_host, g1_group_ids_host);
    } else {
      st1 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_select_epilogue<
          1, true, 128, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
          grouped_g1_tma_epilogue, grouped_arg_workspace, 32ull * 1024ull * 1024ull,
          grouped_gemm_workspace,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(g1_w_dev)), tc_a_scale_dev_,
          const_cast<float*>(g1_s_dev), tc_g1_f32_dev_,
          const_cast<int*>(expert_offsets_dev), total_rows, gemm1_out_, hidden_,
          g1_num_groups, stream, g1_offsets_host, g1_group_ids_host);
    }
  }
  if (st1 != cudaSuccess) return false;

  const char* env_compare_grouped_g1 = std::getenv("FIB_MOE_TC_COMPARE_GROUPED_G1");
  if (!grouped_g1_f16_output && env_compare_grouped_g1 != nullptr &&
      env_compare_grouped_g1[0] == '1') {
    EnsureWorkspace(total_rows, stream);
    for (int le = 0; le < 32; ++le) {
      int n_rows = expert_counts_host[le];
      if (n_rows <= 0) continue;
      int start = expert_offsets_host[le];
      size_t w13_elems = static_cast<size_t>(gemm1_out_) * static_cast<size_t>(hidden_);
      size_t w13s_elems = static_cast<size_t>(gemm1_out_blocks_) * static_cast<size_t>(hidden_blocks_);
      const FP8* w13_e = reinterpret_cast<const FP8*>(gemm1_w_dev + static_cast<size_t>(le) * w13_elems);
      const float* s13_e = gemm1_s_dev + static_cast<size_t>(le) * w13s_elems;
      int w13_scale_elems = gemm1_out_blocks_ * hidden_blocks_;
      copy_scale_nblock_kblock_kernel<<<(w13_scale_elems + kThreads - 1) / kThreads, kThreads,
                                        0, stream>>>(s13_e, w13_scale_elems, tc_b_scale_dev_);
      cudaError_t st_ref = (n_rows >= 256)
          ? launch_cutlass_blockscaled_group_gemm_sm100<1, true, 256, 2,
                                                        cutlass::float_e4m3_t,
                                                        cutlass::float_e4m3_t, float>(
                tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
                32ull * 1024ull * 1024ull,
                reinterpret_cast<FP8*>(tc_a_fp8_dev_ + static_cast<int64_t>(start) * hidden_),
                const_cast<FP8*>(w13_e),
                tc_a_scale_dev_ + static_cast<int64_t>(start) * hidden_blocks_,
                tc_b_scale_dev_, g1_dev_, n_rows, gemm1_out_, hidden_, stream)
          : launch_cutlass_blockscaled_group_gemm_sm100<1, true, 128, 1,
                                                        cutlass::float_e4m3_t,
                                                        cutlass::float_e4m3_t, float>(
                tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
                32ull * 1024ull * 1024ull,
                reinterpret_cast<FP8*>(tc_a_fp8_dev_ + static_cast<int64_t>(start) * hidden_),
                const_cast<FP8*>(w13_e),
                tc_a_scale_dev_ + static_cast<int64_t>(start) * hidden_blocks_,
                tc_b_scale_dev_, g1_dev_, n_rows, gemm1_out_, hidden_, stream);
      if (st_ref != cudaSuccess) continue;
      float* max_diff_dev = nullptr;
      cudaMalloc(&max_diff_dev, sizeof(float));
      cudaMemsetAsync(max_diff_dev, 0, sizeof(float), stream);
      int64_t g1_elems = static_cast<int64_t>(n_rows) * gemm1_out_;
      float* grouped_g1_cur_dev = reinterpret_cast<float*>(tc_float_workspace_dev_);
      cudaMemcpyAsync(grouped_g1_cur_dev,
                      tc_g1_f32_dev_ + static_cast<int64_t>(start) * gemm1_out_,
                      g1_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream);
      compare_abs_diff_kernel<<<static_cast<int>((g1_elems + kThreads - 1) / kThreads),
                                kThreads, 0, stream>>>(
          grouped_g1_cur_dev, g1_dev_, g1_elems, max_diff_dev);
      float max_diff = 0.0f;
      cudaMemcpyAsync(&max_diff, max_diff_dev, sizeof(float), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      cudaFree(max_diff_dev);
      std::fprintf(stderr, "[mxfp] grouped_g1_compare le=%d n_rows=%d max_abs_diff=%g\n",
                   le, n_rows, max_diff);
      const char* env_single_group_debug = std::getenv("FIB_MOE_TC_GROUPED_G1_SINGLE_DEBUG");
      if (le == 0 && start == 0 && env_single_group_debug != nullptr &&
          env_single_group_debug[0] == '1') {
        cudaError_t st_group1 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
            1, true, 128, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
            grouped_arg_workspace, 32ull * 1024ull * 1024ull, grouped_gemm_workspace,
            32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
            const_cast<FP8*>(reinterpret_cast<const FP8*>(gemm1_w_dev)), tc_a_scale_dev_,
            const_cast<float*>(gemm1_s_dev), tc_g1_f32_dev_, const_cast<int*>(expert_offsets_dev),
            n_rows, gemm1_out_, hidden_, 1, stream, expert_offsets_host);
        if (st_group1 == cudaSuccess) {
          cudaMalloc(&max_diff_dev, sizeof(float));
          cudaMemsetAsync(max_diff_dev, 0, sizeof(float), stream);
          compare_abs_diff_kernel<<<static_cast<int>((g1_elems + kThreads - 1) / kThreads),
                                    kThreads, 0, stream>>>(
              tc_g1_f32_dev_, g1_dev_, g1_elems, max_diff_dev);
          max_diff = 0.0f;
          cudaMemcpyAsync(&max_diff, max_diff_dev, sizeof(float), cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);
          cudaFree(max_diff_dev);
          std::fprintf(stderr,
                       "[mxfp] grouped_g1_single_group_compare le=%d n_rows=%d max_abs_diff=%g\n",
                       le, n_rows, max_diff);
        } else {
          std::fprintf(stderr, "[mxfp] grouped_g1_single_group launch failed: %s\n",
                       cudaGetErrorString(st_group1));
        }
      }
      break;
    }
  }

  const bool grouped_g2_after_g1 = grouped_g2_after_g1_candidate;
  if (grouped_g2_after_g1 && padded_total_rows > 0) {
    const char* env_device_padded_offsets = std::getenv("FIB_MOE_TC_DEVICE_PADDED_OFFSETS");
    if (env_device_padded_offsets != nullptr && env_device_padded_offsets[0] == '1') {
      build_padded_offsets_from_expert_offsets_kernel<<<1, 1, 0, stream>>>(
          expert_offsets_dev, tc_m_indptr_dev_);
    } else {
      cudaMemcpyAsync(tc_m_indptr_dev_, padded_offsets, 33 * sizeof(int),
                      cudaMemcpyHostToDevice, stream);
    }

    {
      const char* env_grouped_swiglu = std::getenv("FIB_MOE_TC_GROUPED_SWIGLU");
      const bool grouped_swiglu =
          env_grouped_swiglu == nullptr || env_grouped_swiglu[0] != '0';
      if (grouped_g1_f16_output && !grouped_swiglu) return false;
      if (grouped_swiglu) {
        int col_blocks_per_cta =
            select_grouped_swiglu_col_blocks_per_cta(total_rows, padded_total_rows);
        if (col_blocks_per_cta == 4) {
          dim3 qgrid(padded_total_rows, (intermediate_blocks_ + 3) / 4);
          if (grouped_g1_f16_output) {
            grouped_swiglu_quantize_to_fp8_col_blocks_per_cta_kernel<4, uint16_t, false>
                <<<qgrid, 512, 4 * 128 * sizeof(float), stream>>>(
                    reinterpret_cast<const uint16_t*>(tc_g1_f32_dev_), intermediate_,
                    padded_total_rows, expert_offsets_dev, tc_m_indptr_dev_, tc_c_fp8_dev_,
                    tc_c_scale_dev_);
          } else {
            grouped_swiglu_quantize_to_fp8_col_blocks_per_cta_kernel<4, float, false>
                <<<qgrid, 512, 4 * 128 * sizeof(float), stream>>>(
                    tc_g1_f32_dev_, intermediate_, padded_total_rows, expert_offsets_dev,
                    tc_m_indptr_dev_, tc_c_fp8_dev_, tc_c_scale_dev_);
          }
        } else if (col_blocks_per_cta == 2) {
          dim3 qgrid(padded_total_rows, (intermediate_blocks_ + 1) / 2);
          if (grouped_g1_f16_output) {
            grouped_swiglu_quantize_to_fp8_col_blocks_per_cta_kernel<2, uint16_t, false>
                <<<qgrid, 256, 2 * 128 * sizeof(float), stream>>>(
                    reinterpret_cast<const uint16_t*>(tc_g1_f32_dev_), intermediate_,
                    padded_total_rows, expert_offsets_dev, tc_m_indptr_dev_, tc_c_fp8_dev_,
                    tc_c_scale_dev_);
          } else {
            grouped_swiglu_quantize_to_fp8_col_blocks_per_cta_kernel<2, float, false>
                <<<qgrid, 256, 2 * 128 * sizeof(float), stream>>>(
                    tc_g1_f32_dev_, intermediate_, padded_total_rows, expert_offsets_dev,
                    tc_m_indptr_dev_, tc_c_fp8_dev_, tc_c_scale_dev_);
          }
        } else {
          int rows_per_cta = select_grouped_swiglu_rows_per_cta(total_rows, padded_total_rows);
          dim3 qgrid((padded_total_rows + rows_per_cta - 1) / rows_per_cta,
                     intermediate_blocks_);
          int row_map_min_rows =
              std::max(0, env_int_or_default("FIB_MOE_TC_GROUPED_SWIGLU_ROW_MAP_MIN_ROWS", 4096));
          bool use_row_map =
              rows_per_cta == 1 && row_map_min_rows > 0 && padded_total_rows >= row_map_min_rows;
          if (rows_per_cta == 4) {
            if (grouped_g1_f16_output) {
              grouped_swiglu_quantize_to_fp8_rows_per_cta_kernel<4, uint16_t, false>
                  <<<qgrid, 128, 4 * 128 * sizeof(float), stream>>>(
                      reinterpret_cast<const uint16_t*>(tc_g1_f32_dev_), intermediate_,
                      padded_total_rows, expert_offsets_dev, tc_m_indptr_dev_, tc_c_fp8_dev_,
                      tc_c_scale_dev_);
            } else {
              grouped_swiglu_quantize_to_fp8_rows_per_cta_kernel<4, float, false>
                  <<<qgrid, 128, 4 * 128 * sizeof(float), stream>>>(
                      tc_g1_f32_dev_, intermediate_, padded_total_rows, expert_offsets_dev,
                      tc_m_indptr_dev_, tc_c_fp8_dev_, tc_c_scale_dev_);
            }
          } else if (rows_per_cta == 2) {
            if (grouped_g1_f16_output) {
              grouped_swiglu_quantize_to_fp8_rows_per_cta_kernel<2, uint16_t, false>
                  <<<qgrid, 128, 2 * 128 * sizeof(float), stream>>>(
                      reinterpret_cast<const uint16_t*>(tc_g1_f32_dev_), intermediate_,
                      padded_total_rows, expert_offsets_dev, tc_m_indptr_dev_, tc_c_fp8_dev_,
                      tc_c_scale_dev_);
            } else {
              grouped_swiglu_quantize_to_fp8_rows_per_cta_kernel<2, float, false>
                  <<<qgrid, 128, 2 * 128 * sizeof(float), stream>>>(
                      tc_g1_f32_dev_, intermediate_, padded_total_rows, expert_offsets_dev,
                      tc_m_indptr_dev_, tc_c_fp8_dev_, tc_c_scale_dev_);
            }
          } else if (use_row_map) {
            build_padded_compact_row_map_kernel<<<32, 128, 0, stream>>>(
                expert_offsets_dev, tc_m_indptr_dev_, tc_padded_compact_rows_dev_);
            if (grouped_g1_f16_output) {
              grouped_swiglu_quantize_to_fp8_mapped_kernel<uint16_t, false>
                  <<<qgrid, 128, 4 * sizeof(float), stream>>>(
                      reinterpret_cast<const uint16_t*>(tc_g1_f32_dev_), intermediate_,
                      padded_total_rows, tc_padded_compact_rows_dev_, tc_c_fp8_dev_,
                      tc_c_scale_dev_);
            } else {
              grouped_swiglu_quantize_to_fp8_mapped_kernel<float, false>
                  <<<qgrid, 128, 4 * sizeof(float), stream>>>(
                      tc_g1_f32_dev_, intermediate_, padded_total_rows,
                      tc_padded_compact_rows_dev_, tc_c_fp8_dev_, tc_c_scale_dev_);
            }
          } else {
            if (grouped_g1_f16_output) {
              grouped_swiglu_quantize_to_fp8_kernel<uint16_t, false>
                  <<<qgrid, 128, 4 * sizeof(float), stream>>>(
                      reinterpret_cast<const uint16_t*>(tc_g1_f32_dev_), intermediate_,
                      padded_total_rows, expert_offsets_dev, tc_m_indptr_dev_, tc_c_fp8_dev_,
                      tc_c_scale_dev_);
            } else {
              grouped_swiglu_quantize_to_fp8_kernel<float, false>
                  <<<qgrid, 128, 4 * sizeof(float), stream>>>(
                      tc_g1_f32_dev_, intermediate_, padded_total_rows, expert_offsets_dev,
                      tc_m_indptr_dev_, tc_c_fp8_dev_, tc_c_scale_dev_);
            }
          }
        }
      } else {
        for (int le = 0; le < 32; ++le) {
          int n_rows = expert_counts_host[le];
          if (n_rows == 0) continue;
          int start = expert_offsets_host[le];
          int padded_start = padded_offsets[le];
          int padded_rows = (n_rows + 3) & ~3;
          int swiglu_rows_per_cta = select_swiglu_rows_per_cta(total_rows, padded_rows);
          dim3 qgrid((padded_rows + swiglu_rows_per_cta - 1) / swiglu_rows_per_cta,
                     intermediate_blocks_);
          float* g1_ptr = tc_g1_f32_dev_ + static_cast<int64_t>(start) * gemm1_out_;
          uint8_t* c_ptr = tc_c_fp8_dev_ + static_cast<int64_t>(padded_start) * intermediate_;
          float* c_scale_ptr =
              tc_c_scale_dev_ + static_cast<int64_t>(padded_start) * intermediate_blocks_;
          if (swiglu_rows_per_cta == 4) {
            swiglu_quantize_float_to_fp8_rows_per_cta_kernel<4>
                <<<qgrid, 128, 4 * 128 * sizeof(float), stream>>>(
                    g1_ptr, intermediate_, n_rows, padded_rows, c_ptr, c_scale_ptr, true);
          } else if (swiglu_rows_per_cta == 2) {
            swiglu_quantize_float_to_fp8_rows_per_cta_kernel<2>
                <<<qgrid, 128, 2 * 128 * sizeof(float), stream>>>(
                    g1_ptr, intermediate_, n_rows, padded_rows, c_ptr, c_scale_ptr, true);
          } else {
            swiglu_quantize_float_to_fp8_kernel<<<qgrid, 128, 128 * sizeof(float), stream>>>(
                g1_ptr, intermediate_, n_rows, padded_rows, c_ptr, c_scale_ptr, true);
          }
        }
      }
    }

    TcGemmDispatch g2_dispatch = select_trtllm_like_gemm2_dispatch(total_rows, 64);
    int g2_tile_m = select_tile_m_from_env("FIB_MOE_GROUPED_G2_TILE_M", g2_dispatch.tile_m);
    int g2_mma_sm = select_mma_sm_from_env("FIB_MOE_GROUPED_G2_MMA_SM", g2_dispatch.mma_sms);
    cudaError_t st_g2 = cudaSuccess;
    if (g2_mma_sm == 2) {
      st_g2 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
          1, true, 256, 2, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
          tc_group_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_group_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(gemm2_w_dev)), tc_c_scale_dev_,
          const_cast<float*>(gemm2_s_dev), tc_d_f32_dev_, tc_m_indptr_dev_, padded_total_rows,
          hidden_, intermediate_, 32, stream, device_group_args ? nullptr : padded_offsets);
    } else if (g2_tile_m == 128) {
      st_g2 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
          1, true, 128, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
          tc_group_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_group_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(gemm2_w_dev)), tc_c_scale_dev_,
          const_cast<float*>(gemm2_s_dev), tc_d_f32_dev_, tc_m_indptr_dev_, padded_total_rows,
          hidden_, intermediate_, 32, stream, device_group_args ? nullptr : padded_offsets);
    } else {
      st_g2 = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
          1, true, 64, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
          tc_group_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_group_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_),
          const_cast<FP8*>(reinterpret_cast<const FP8*>(gemm2_w_dev)), tc_c_scale_dev_,
          const_cast<float*>(gemm2_s_dev), tc_d_f32_dev_, tc_m_indptr_dev_, padded_total_rows,
          hidden_, intermediate_, 32, stream, device_group_args ? nullptr : padded_offsets);
    }
    if (st_g2 == cudaSuccess) {
      if (routed_positions_dev == nullptr || routed_local_experts_dev == nullptr ||
          routed_weights_dev == nullptr || output_dev == nullptr) {
        return false;
      }
      WriteTcPaddedScratchToBf16Output(t, expert_offsets_dev, tc_m_indptr_dev_,
                                       routed_positions_dev, routed_local_experts_dev,
                                       routed_weights_dev, output_dev, stream);
      return true;
    }
  }

  for (int le = 0; le < 32; ++le) {
    int n_rows = expert_counts_host[le];
    if (n_rows == 0) continue;
    int start = expert_offsets_host[le];
    // Match the direct-output gather scratch convention: compact row position
    // plus four padding rows per expert. This keeps grouped G1 compatible with
    // the existing top-k gather and avoids an out_acc/final-convert pass.
    int scratch_start = direct_output ? (start + 4 * le) : padded_offsets[le];
    const float* permuted_w_e = direct_output ? nullptr : (permuted_w_dev + start);
    RunExpertGemm2TcFromG1(t, total_rows, n_rows, start, scratch_start, permuted_tok_dev + start,
                           permuted_w_e, le, gemm2_w_dev, gemm2_s_dev, out_acc_dev,
                           stream, !direct_output);
  }
  if (direct_output) {
    if (routed_positions_dev == nullptr || routed_local_experts_dev == nullptr ||
        routed_weights_dev == nullptr || output_dev == nullptr) {
      return false;
    }
    WriteTcScratchToBf16Output(t, routed_positions_dev, routed_local_experts_dev,
                               routed_weights_dev, output_dev, stream);
  }
  return true;
#else
  (void)hidden_fp8_dev;
  (void)hidden_scale_dev;
  (void)t;
  (void)total_rows;
  (void)expert_offsets_dev;
  (void)expert_counts_host;
  (void)expert_offsets_host;
  (void)permuted_tok_dev;
  (void)permuted_w_dev;
  (void)gemm1_w_dev;
  (void)gemm1_s_dev;
  (void)gemm2_w_dev;
  (void)gemm2_s_dev;
  (void)out_acc_dev;
  (void)stream;
  (void)direct_output;
  (void)routed_positions_dev;
  (void)routed_local_experts_dev;
  (void)routed_weights_dev;
  (void)output_dev;
  return false;
#endif
}

bool DeviceMxfpGemmModule::RunExpertGemm1ThenGroupedGemm2Tc(
    const uint8_t* hidden_fp8_dev, const float* hidden_scale_dev, int64_t t, int total_rows,
    const int* expert_offsets_dev, const int* expert_counts_host,
    const int* expert_offsets_host, const int* permuted_tok_dev,
    const int* permuted_expert_dev, const float* permuted_w_dev, const uint8_t* gemm1_w_dev,
    const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
    const float* gemm2_s_dev, float* out_acc_dev, cudaStream_t stream) {
#if FIB_HAS_DIRECT_CUTLASS_SM100
  const char* env_enable = std::getenv("FIB_MOE_TC_ENABLE_GROUPED_G2");
  if (env_enable == nullptr || env_enable[0] != '1') return false;
  if (total_rows <= 0) return true;
  if (!SupportsTcPath()) return false;

  std::array<int, 33> padded_offsets_fallback{};
  int* padded_offsets =
      tc_m_indptr_host_ == nullptr ? padded_offsets_fallback.data() : tc_m_indptr_host_;
  padded_offsets[0] = 0;
  for (int le = 0; le < 32; ++le) {
    padded_offsets[le + 1] = padded_offsets[le] + ((expert_counts_host[le] + 3) & ~3);
  }
  int padded_total_rows = padded_offsets[32];
  if (padded_total_rows <= 0) return true;

  EnsureTcWorkspace(padded_total_rows);
  for (int le = 0; le < 32; ++le) {
    int n_rows = expert_counts_host[le];
    if (n_rows == 0) continue;
    int start = expert_offsets_host[le];
    int padded_start = padded_offsets[le];
    RunExpertPermutedTcToScratch(hidden_fp8_dev, hidden_scale_dev, t, padded_total_rows,
                                 padded_start, n_rows, permuted_tok_dev + start, le,
                                 gemm1_w_dev, gemm1_s_dev, gemm2_w_dev, gemm2_s_dev, stream,
                                 false);
  }

  cudaMemcpyAsync(tc_m_indptr_dev_, padded_offsets, 33 * sizeof(int), cudaMemcpyHostToDevice,
                  stream);

  using FP8 = cutlass::float_e4m3_t;
  TcGemmDispatch g2_dispatch = select_trtllm_like_gemm2_dispatch(total_rows, 64);
  int g2_tile_m = select_tile_m_from_env("FIB_MOE_GROUPED_G2_TILE_M", g2_dispatch.tile_m);
  int g2_mma_sm = select_mma_sm_from_env("FIB_MOE_GROUPED_G2_MMA_SM", g2_dispatch.mma_sms);
  cudaError_t st = cudaSuccess;
  if (g2_mma_sm == 2) {
    st = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
        1, true, 256, 2, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_),
        const_cast<FP8*>(reinterpret_cast<const FP8*>(gemm2_w_dev)), tc_c_scale_dev_,
        const_cast<float*>(gemm2_s_dev), tc_d_f32_dev_, tc_m_indptr_dev_, padded_total_rows,
        hidden_, intermediate_, 32, stream, padded_offsets);
  } else if (g2_tile_m == 128) {
    st = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
        1, true, 128, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_),
        const_cast<FP8*>(reinterpret_cast<const FP8*>(gemm2_w_dev)), tc_c_scale_dev_,
        const_cast<float*>(gemm2_s_dev), tc_d_f32_dev_, tc_m_indptr_dev_, padded_total_rows,
        hidden_, intermediate_, 32, stream, padded_offsets);
  } else {
    st = launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
        1, true, 64, 1, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_),
        const_cast<FP8*>(reinterpret_cast<const FP8*>(gemm2_w_dev)), tc_c_scale_dev_,
        const_cast<float*>(gemm2_s_dev), tc_d_f32_dev_, tc_m_indptr_dev_, padded_total_rows,
        hidden_, intermediate_, 32, stream, padded_offsets);
  }
  if (st != cudaSuccess) return false;

  constexpr int kThreads = 256;
  dim3 grid(total_rows, (hidden_ + kThreads - 1) / kThreads);
  scatter_all_float_weighted_by_padded_expert_row_kernel<<<grid, kThreads, 0, stream>>>(
      tc_d_f32_dev_, hidden_, total_rows, expert_offsets_dev, tc_m_indptr_dev_, permuted_tok_dev,
      permuted_expert_dev, permuted_w_dev, out_acc_dev);
  return true;
#else
  (void)hidden_fp8_dev;
  (void)hidden_scale_dev;
  (void)t;
  (void)total_rows;
  (void)expert_offsets_dev;
  (void)expert_counts_host;
  (void)expert_offsets_host;
  (void)permuted_tok_dev;
  (void)permuted_expert_dev;
  (void)permuted_w_dev;
  (void)gemm1_w_dev;
  (void)gemm1_s_dev;
  (void)gemm2_w_dev;
  (void)gemm2_s_dev;
  (void)out_acc_dev;
  (void)stream;
  return false;
#endif
}

void DeviceMxfpGemmModule::RunExpertPermutedTc(const uint8_t* hidden_fp8_dev,
                                               const float* hidden_scale_dev, int64_t t,
                                               int n_rows, const int* permuted_tok_e,
                                               const float* permuted_w_e, int local_expert_idx,
                                               const uint8_t* gemm1_w_dev,
                                               const float* gemm1_s_dev,
                                               const uint8_t* gemm2_w_dev,
                                               const float* gemm2_s_dev,
                                               float* out_acc_dev, cudaStream_t stream) {
  if (n_rows <= 0) return;
  if (!SupportsTcPath()) {
    return RunExpertPermuted(nullptr, t, n_rows, permuted_tok_e, permuted_w_e, local_expert_idx,
                             gemm1_w_dev, gemm1_s_dev, gemm2_w_dev, gemm2_s_dev, out_acc_dev,
                             stream);
  }

#if FIB_HAS_DIRECT_CUTLASS_SM100
  EnsureTcWorkspace(n_rows);
  int padded_rows = (n_rows + 3) & ~3;

  size_t w13_elems = static_cast<size_t>(gemm1_out_) * static_cast<size_t>(hidden_);
  size_t w13s_elems = static_cast<size_t>(gemm1_out_blocks_) * static_cast<size_t>(hidden_blocks_);
  size_t w2_elems = static_cast<size_t>(hidden_) * static_cast<size_t>(intermediate_);
  size_t w2s_elems = static_cast<size_t>(hidden_blocks_) * static_cast<size_t>(intermediate_blocks_);

  const uint8_t* w13_e = gemm1_w_dev + static_cast<size_t>(local_expert_idx) * w13_elems;
  const float* s13_e = gemm1_s_dev + static_cast<size_t>(local_expert_idx) * w13s_elems;
  const uint8_t* w2_e = gemm2_w_dev + static_cast<size_t>(local_expert_idx) * w2_elems;
  const float* s2_e = gemm2_s_dev + static_cast<size_t>(local_expert_idx) * w2s_elems;
  // Standalone contract probing shows the direct CUTLASS launcher expects the
  // raw [N, K] payload as-is here. Keep transpose only as an explicit debug
  // override.
  const char* env_force_transpose_b = std::getenv("FIB_MOE_TC_FORCE_TRANSPOSE_B");
  const bool transpose_b =
      (env_force_transpose_b != nullptr && env_force_transpose_b[0] == '1');
  const char* env_scale_major_k = std::getenv("FIB_MOE_TC_SCALE_MAJOR_K");
  const bool scale_major_k = (env_scale_major_k == nullptr || env_scale_major_k[0] == '1');
  constexpr int c_scale_gran_m = 1;

  using FP8 = cutlass::float_e4m3_t;

  constexpr int kThreads = 256;
  int hidden_chunks16 = hidden_ / 16;
  bool vec16_gather = (hidden_ % 16 == 0) && use_vec16_hidden_gather(padded_rows);
  int64_t a_elems =
      static_cast<int64_t>(padded_rows) * (vec16_gather ? hidden_chunks16 : hidden_);
  int64_t a_scale_elems = static_cast<int64_t>(padded_rows) * hidden_blocks_;
  if (scale_major_k) {
    if (vec16_gather) {
      gather_hidden_fp8_vec16_and_scale_rows_mn_major_kernel<<<
          static_cast<int>((std::max(a_elems, a_scale_elems) + kThreads - 1) / kThreads),
          kThreads, 0, stream>>>(hidden_fp8_dev, hidden_scale_dev, permuted_tok_e, t, n_rows,
                                 padded_rows, hidden_chunks16, hidden_, hidden_blocks_,
                                 tc_a_fp8_dev_, tc_a_scale_dev_);
    } else {
      gather_hidden_fp8_and_scale_rows_mn_major_kernel<<<
          static_cast<int>((std::max(a_elems, a_scale_elems) + kThreads - 1) / kThreads),
          kThreads, 0, stream>>>(hidden_fp8_dev, hidden_scale_dev, permuted_tok_e, t, n_rows,
                                 padded_rows, hidden_, hidden_blocks_, tc_a_fp8_dev_,
                                 tc_a_scale_dev_);
    }
  } else {
    a_elems = static_cast<int64_t>(padded_rows) * hidden_;
    gather_hidden_fp8_rows_kernel<<<static_cast<int>((a_elems + kThreads - 1) / kThreads),
                                    kThreads, 0, stream>>>(
        hidden_fp8_dev, permuted_tok_e, n_rows, padded_rows, hidden_, tc_a_fp8_dev_);
    gather_hidden_scale_rows_kernel<<<static_cast<int>((a_scale_elems + kThreads - 1) / kThreads),
                                      kThreads, 0, stream>>>(
        hidden_scale_dev, permuted_tok_e, t, n_rows, padded_rows, hidden_blocks_,
        tc_a_scale_dev_);
  }

  const int small_rows_threshold =
      std::max(0, env_int_or_default("FIB_MOE_TC_SMALL_ROWS", 0));
  if (n_rows <= small_rows_threshold) {
    EnsureWorkspace(n_rows, stream);
    float* a_compact_dev = reinterpret_cast<float*>(tc_int_workspace_dev_);
    int64_t a_f32_elems = static_cast<int64_t>(n_rows) * hidden_;
    int64_t g1_elems = static_cast<int64_t>(n_rows) * gemm1_out_;
    int64_t c_elems = static_cast<int64_t>(n_rows) * intermediate_;
    int64_t out_elems = static_cast<int64_t>(n_rows) * hidden_;
    dequant_fp8_rows_kernel<<<static_cast<int>((a_f32_elems + kThreads - 1) / kThreads),
                              kThreads, 0, stream>>>(
        tc_a_fp8_dev_, tc_a_scale_dev_, n_rows, hidden_, 1, hidden_blocks_, scale_major_k,
        a_compact_dev);
    gemm1_compact_kernel<<<static_cast<int>((g1_elems + kThreads - 1) / kThreads), kThreads, 0,
                           stream>>>(
        a_compact_dev, hidden_, gemm1_out_, block_, hidden_blocks_, n_rows, w13_e, s13_e,
        emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_, g1_dev_);
    swiglu_permuted_kernel<<<static_cast<int>((c_elems + kThreads - 1) / kThreads), kThreads, 0,
                             stream>>>(g1_dev_, intermediate_, n_rows, false, c_dev_);
    gemm2_scatter_accumulate_kernel<<<static_cast<int>((out_elems + kThreads - 1) / kThreads),
                                      kThreads, 0, stream>>>(
        c_dev_, hidden_, intermediate_, block_, intermediate_blocks_, n_rows, permuted_tok_e,
        permuted_w_e, w2_e, s2_e, emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_,
        out_acc_dev);
    return;
  }

  int64_t w13_transpose_elems = static_cast<int64_t>(gemm1_out_) * hidden_;
  if (transpose_b) {
    transpose_rowmajor_nk_to_colmajor_nk_kernel<<<
        static_cast<int>((w13_transpose_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        w13_e, gemm1_out_, hidden_, tc_b_col_dev_);
  }
  FP8* w13_tc = transpose_b ? reinterpret_cast<FP8*>(tc_b_col_dev_)
                            : const_cast<FP8*>(reinterpret_cast<const FP8*>(w13_e));
  int w13_scale_elems = gemm1_out_blocks_ * hidden_blocks_;
  float* w13_scale_tc = const_cast<float*>(s13_e);
  if (scale_major_k) {
    (void)w13_scale_elems;
  } else {
    transpose_scale_nblock_kblock_to_kblock_nblock_kernel<<<
        (w13_scale_elems + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        s13_e, gemm1_out_blocks_, hidden_blocks_, tc_b_scale_dev_);
    w13_scale_tc = tc_b_scale_dev_;
  }

  auto run_gemm1_1sm = [&]() {
    TcGemmDispatch g1_dispatch = select_trtllm_like_gemm1_dispatch(n_rows, padded_rows);
    int g1_tile_m = select_tile_m_from_env("FIB_MOE_G1_TILE_M", g1_dispatch.tile_m);
    const char* env_g1_tn = std::getenv("FIB_MOE_G1_TRANSPOSED");
    if (env_g1_tn != nullptr && env_g1_tn[0] == '1' && scale_major_k && !transpose_b) {
      int g1_tile_n = select_tile_m_from_env("FIB_MOE_G1_TN_TILE_N", 64);
      if (g1_tile_n == 128) {
        return launch_cutlass_blockscaled_group_gemm_tn_sm100<128, 1, true, 128,
                                                              cutlass::float_e4m3_t,
                                                              cutlass::float_e4m3_t, float>(
            tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
            32ull * 1024ull * 1024ull, const_cast<FP8*>(reinterpret_cast<const FP8*>(w13_e)),
            reinterpret_cast<FP8*>(tc_a_fp8_dev_), w13_scale_tc, tc_a_scale_dev_,
            tc_g1_f32_dev_, gemm1_out_, padded_rows, hidden_, stream);
      }
      return launch_cutlass_blockscaled_group_gemm_tn_sm100<128, 1, true, 64,
                                                            cutlass::float_e4m3_t,
                                                            cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, const_cast<FP8*>(reinterpret_cast<const FP8*>(w13_e)),
          reinterpret_cast<FP8*>(tc_a_fp8_dev_), w13_scale_tc, tc_a_scale_dev_,
          tc_g1_f32_dev_, gemm1_out_, padded_rows, hidden_, stream);
    }
    if (scale_major_k) {
      if (g1_tile_m == 64) {
        return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 64, 1,
                                                           cutlass::float_e4m3_t,
                                                           cutlass::float_e4m3_t, float>(
            tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
            32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_), w13_tc,
            tc_a_scale_dev_, w13_scale_tc, tc_g1_f32_dev_,
            padded_rows, gemm1_out_, hidden_, stream);
      }
      return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 128, 1,
                                                         cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_), w13_tc,
          tc_a_scale_dev_, w13_scale_tc, tc_g1_f32_dev_,
          padded_rows, gemm1_out_, hidden_, stream);
    }
    if (g1_tile_m == 64) {
      return launch_cutlass_blockscaled_group_gemm_sm100<1, false, 64, 1,
                                                         cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_), w13_tc,
          tc_a_scale_dev_, w13_scale_tc, tc_g1_f32_dev_,
          padded_rows, gemm1_out_, hidden_, stream);
    }
    return launch_cutlass_blockscaled_group_gemm_sm100<1, false, 128, 1,
                                                       cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_), w13_tc,
        tc_a_scale_dev_, w13_scale_tc, tc_g1_f32_dev_,
        padded_rows, gemm1_out_, hidden_, stream);
  };
  auto run_gemm1_2sm = [&]() {
    if (scale_major_k) {
      return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 256, 2,
                                                         cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_), w13_tc,
          tc_a_scale_dev_, w13_scale_tc, tc_g1_f32_dev_,
          padded_rows, gemm1_out_, hidden_, stream);
    }
    return launch_cutlass_blockscaled_group_gemm_sm100<1, false, 256, 2,
                                                       cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_), w13_tc,
        tc_a_scale_dev_, w13_scale_tc, tc_g1_f32_dev_,
        padded_rows, gemm1_out_, hidden_, stream);
  };
  TcGemmDispatch g1_dispatch = select_trtllm_like_gemm1_dispatch(n_rows, padded_rows);
  int g1_mma_sm = select_mma_sm_from_env("FIB_MOE_G1_MMA_SM", g1_dispatch.mma_sms);
  cudaError_t st1 = (g1_mma_sm == 2) ? run_gemm1_2sm() : run_gemm1_1sm();
  if (st1 != cudaSuccess) return;

  const char* env_compare_g1 = std::getenv("FIB_MOE_TC_COMPARE_G1");
  if (env_compare_g1 != nullptr && env_compare_g1[0] == '1') {
    EnsureWorkspace(n_rows, stream);
    int64_t g1_elems = static_cast<int64_t>(n_rows) * gemm1_out_;
    int64_t g1_bytes = g1_elems * static_cast<int64_t>(sizeof(float));
    int64_t a_elems_f32 = static_cast<int64_t>(n_rows) * hidden_;
    int64_t a_bytes_f32 = a_elems_f32 * static_cast<int64_t>(sizeof(float));
    if (a_bytes_f32 + g1_bytes + static_cast<int64_t>(sizeof(float)) <= 32ll * 1024ll * 1024ll) {
      float* a_ref_dev = reinterpret_cast<float*>(tc_int_workspace_dev_);
      float* g1_ref_dev = reinterpret_cast<float*>(tc_float_workspace_dev_);
      float* g1_max_diff_dev =
          reinterpret_cast<float*>(reinterpret_cast<char*>(tc_float_workspace_dev_) + g1_bytes);
      dequant_fp8_rows_kernel<<<static_cast<int>((a_elems_f32 + kThreads - 1) / kThreads), kThreads,
                                0, stream>>>(tc_a_fp8_dev_, tc_a_scale_dev_, n_rows, hidden_, 1,
                                             hidden_blocks_, scale_major_k, a_ref_dev);
      gemm1_compact_kernel<<<static_cast<int>((g1_elems + kThreads - 1) / kThreads), kThreads, 0,
                             stream>>>(
          a_ref_dev, hidden_, gemm1_out_, block_, hidden_blocks_, n_rows, w13_e, s13_e,
          emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_, g1_ref_dev);
      cudaMemcpyAsync(g1_dev_, tc_g1_f32_dev_, g1_elems * sizeof(float), cudaMemcpyDeviceToDevice,
                      stream);
      cudaMemsetAsync(g1_max_diff_dev, 0, sizeof(float), stream);
      compare_abs_diff_kernel<<<static_cast<int>((g1_elems + kThreads - 1) / kThreads), kThreads,
                                0, stream>>>(tc_g1_f32_dev_, g1_ref_dev, g1_elems,
                                             g1_max_diff_dev);
      float g1_max_diff = 0.0f;
      cudaMemcpyAsync(&g1_max_diff, g1_max_diff_dev, sizeof(float), cudaMemcpyDeviceToHost,
                      stream);
      cudaStreamSynchronize(stream);
      std::fprintf(stderr, "[mxfp] local_expert=%d n_rows=%d g1_max_abs_diff=%g\n",
                   local_expert_idx, n_rows, g1_max_diff);
      std::fflush(stderr);
    }
  }

  if (std::getenv("FIB_MOE_TC_GEMM1_ONLY") != nullptr) {
    EnsureWorkspace(n_rows, stream);
    int64_t c_elems = static_cast<int64_t>(n_rows) * intermediate_;
    int64_t out_elems = static_cast<int64_t>(n_rows) * hidden_;
    swiglu_permuted_kernel<<<static_cast<int>((c_elems + kThreads - 1) / kThreads),
                              kThreads, 0, stream>>>(
        tc_g1_f32_dev_, intermediate_, n_rows, false, c_dev_);
    gemm2_scatter_accumulate_kernel<<<static_cast<int>((out_elems + kThreads - 1) / kThreads),
                                      kThreads, 0, stream>>>(
        c_dev_, hidden_, intermediate_, block_, intermediate_blocks_, n_rows, permuted_tok_e,
        permuted_w_e, w2_e, s2_e, emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_,
        out_acc_dev);
    return;
  }

  if (tc_fp16_middle_) {
    dim3 qgrid(padded_rows, intermediate_blocks_);
    uint16_t* g1_f16_ptr = reinterpret_cast<uint16_t*>(tc_float_workspace_dev_);
    float_to_f16_kernel<<<
        static_cast<int>(((static_cast<int64_t>(padded_rows) * gemm1_out_) + kThreads - 1) /
                         kThreads),
        kThreads, 0, stream>>>(tc_g1_f32_dev_,
                               static_cast<int64_t>(padded_rows) * gemm1_out_, g1_f16_ptr);
    swiglu_quantize_f16_to_fp8_kernel<<<qgrid, 128, 128 * sizeof(float), stream>>>(
        g1_f16_ptr, intermediate_, n_rows, padded_rows, tc_c_fp8_dev_, tc_c_scale_dev_,
        scale_major_k);
  } else {
    int swiglu_rows_per_cta = select_swiglu_rows_per_cta(n_rows, padded_rows);
    dim3 qgrid((padded_rows + swiglu_rows_per_cta - 1) / swiglu_rows_per_cta,
               intermediate_blocks_);
    if (swiglu_rows_per_cta == 4) {
      swiglu_quantize_float_to_fp8_rows_per_cta_kernel<4>
          <<<qgrid, 128, 4 * 128 * sizeof(float), stream>>>(
              tc_g1_f32_dev_, intermediate_, n_rows, padded_rows, tc_c_fp8_dev_, tc_c_scale_dev_,
              scale_major_k);
    } else if (swiglu_rows_per_cta == 2) {
      swiglu_quantize_float_to_fp8_rows_per_cta_kernel<2>
          <<<qgrid, 128, 2 * 128 * sizeof(float), stream>>>(
              tc_g1_f32_dev_, intermediate_, n_rows, padded_rows, tc_c_fp8_dev_, tc_c_scale_dev_,
              scale_major_k);
    } else {
      swiglu_quantize_float_to_fp8_kernel<<<qgrid, 128, 128 * sizeof(float), stream>>>(
          tc_g1_f32_dev_, intermediate_, n_rows, padded_rows, tc_c_fp8_dev_, tc_c_scale_dev_,
          scale_major_k);
    }
  }
  const char* env_compare_c = std::getenv("FIB_MOE_TC_COMPARE_C");
  if (env_compare_c != nullptr && env_compare_c[0] == '1') {
    EnsureWorkspace(n_rows, stream);
    int64_t c_elems = static_cast<int64_t>(n_rows) * intermediate_;
    swiglu_permuted_kernel<<<static_cast<int>((c_elems + kThreads - 1) / kThreads), kThreads, 0,
                             stream>>>(tc_g1_f32_dev_, intermediate_, n_rows, false, c_dev_);
    int64_t c_all_elems = static_cast<int64_t>(n_rows) * intermediate_;
    int64_t c_all_bytes = c_all_elems * static_cast<int64_t>(sizeof(float));
    int64_t g1_ref_elems = static_cast<int64_t>(n_rows) * gemm1_out_;
    int64_t g1_ref_bytes = g1_ref_elems * static_cast<int64_t>(sizeof(float));
    int64_t a_ref_elems = static_cast<int64_t>(n_rows) * hidden_;
    int64_t a_ref_bytes = a_ref_elems * static_cast<int64_t>(sizeof(float));
    int64_t need_bytes = 2 * c_all_bytes + 2 * static_cast<int64_t>(sizeof(float));
    if (need_bytes <= 32ll * 1024ll * 1024ll &&
        a_ref_bytes + g1_ref_bytes <= 32ll * 1024ll * 1024ll) {
      float* a_ref_dev = reinterpret_cast<float*>(tc_int_workspace_dev_);
      float* g1_ref_dev =
          reinterpret_cast<float*>(reinterpret_cast<char*>(tc_int_workspace_dev_) + a_ref_bytes);
      float* c_ref_dev = reinterpret_cast<float*>(tc_float_workspace_dev_);
      float* c_deq_dev = reinterpret_cast<float*>(reinterpret_cast<char*>(tc_float_workspace_dev_) +
                                                  c_all_bytes);
      float* c_cur_max_diff_dev =
          reinterpret_cast<float*>(reinterpret_cast<char*>(tc_float_workspace_dev_) +
                                   2 * c_all_bytes);
      float* c_q_max_diff_dev =
          reinterpret_cast<float*>(reinterpret_cast<char*>(tc_float_workspace_dev_) +
                                   2 * c_all_bytes + sizeof(float));
      dequant_fp8_rows_kernel<<<static_cast<int>((a_ref_elems + kThreads - 1) / kThreads), kThreads,
                                0, stream>>>(tc_a_fp8_dev_, tc_a_scale_dev_, n_rows, hidden_, 1,
                                             hidden_blocks_, scale_major_k, a_ref_dev);
      gemm1_compact_kernel<<<static_cast<int>((g1_ref_elems + kThreads - 1) / kThreads), kThreads,
                             0, stream>>>(
          a_ref_dev, hidden_, gemm1_out_, block_, hidden_blocks_, n_rows, w13_e, s13_e,
          emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_, g1_ref_dev);
      swiglu_permuted_kernel<<<static_cast<int>((c_all_elems + kThreads - 1) / kThreads), kThreads,
                               0, stream>>>(g1_ref_dev, intermediate_, n_rows, false, c_ref_dev);
      dequant_fp8_rows_native_kernel<<<
          static_cast<int>((c_all_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
          tc_c_fp8_dev_, tc_c_scale_dev_, n_rows, intermediate_, c_scale_gran_m,
          intermediate_blocks_,
          scale_major_k, c_deq_dev);
      cudaMemsetAsync(c_cur_max_diff_dev, 0, sizeof(float), stream);
      cudaMemsetAsync(c_q_max_diff_dev, 0, sizeof(float), stream);
      compare_abs_diff_kernel<<<static_cast<int>((c_all_elems + kThreads - 1) / kThreads), kThreads,
                                0, stream>>>(c_ref_dev, c_dev_, c_all_elems, c_cur_max_diff_dev);
      compare_abs_diff_kernel<<<static_cast<int>((c_all_elems + kThreads - 1) / kThreads), kThreads,
                                0, stream>>>(c_ref_dev, c_deq_dev, c_all_elems, c_q_max_diff_dev);
      float c_cur_max_diff = 0.0f;
      float c_q_max_diff = 0.0f;
      cudaMemcpyAsync(&c_cur_max_diff, c_cur_max_diff_dev, sizeof(float), cudaMemcpyDeviceToHost,
                      stream);
      cudaMemcpyAsync(&c_q_max_diff, c_q_max_diff_dev, sizeof(float), cudaMemcpyDeviceToHost,
                      stream);
      cudaStreamSynchronize(stream);
      std::fprintf(stderr,
                   "[mxfp] local_expert=%d n_rows=%d c_float_max_abs_diff=%g c_quant_max_abs_diff=%g\n",
                   local_expert_idx, n_rows, c_cur_max_diff, c_q_max_diff);
      const char* env_dump_c_block_errors = std::getenv("FIB_MOE_TC_DUMP_C_BLOCK_ERRORS");
      if (env_dump_c_block_errors != nullptr && env_dump_c_block_errors[0] == '1' && n_rows > 0 &&
          c_scale_gran_m == 1) {
        std::vector<float> c_ref_host(static_cast<size_t>(c_all_elems));
        std::vector<float> c_deq_host(static_cast<size_t>(c_all_elems));
        std::vector<float> scale_host(static_cast<size_t>(n_rows) * intermediate_blocks_);
        cudaMemcpyAsync(c_ref_host.data(), c_ref_dev, c_all_bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(c_deq_host.data(), c_deq_dev, c_all_bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(scale_host.data(), tc_c_scale_dev_,
                        scale_host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        struct BlockErr {
          int row;
          int cb;
          float max_abs;
          float mean_abs;
          float scale;
        };
        std::array<BlockErr, 4> top_blocks{};
        for (auto& e : top_blocks) {
          e.row = -1;
          e.cb = -1;
          e.max_abs = -1.0f;
          e.mean_abs = -1.0f;
          e.scale = 0.0f;
        }

        for (int row = 0; row < n_rows; ++row) {
          for (int cb = 0; cb < intermediate_blocks_; ++cb) {
            float max_abs = 0.0f;
            float sum_abs = 0.0f;
            int col0 = cb * 128;
            for (int u = 0; u < 128; ++u) {
              int col = col0 + u;
              float diff = std::fabs(
                  c_ref_host[static_cast<size_t>(row) * intermediate_ + col] -
                  c_deq_host[static_cast<size_t>(row) * intermediate_ + col]);
              max_abs = std::max(max_abs, diff);
              sum_abs += diff;
            }
            BlockErr cur{row, cb, max_abs, sum_abs / 128.0f,
                         scale_major_k ? scale_host[static_cast<size_t>(row) * intermediate_blocks_ + cb]
                                       : scale_host[static_cast<size_t>(cb) * n_rows + row]};
            for (int slot = 0; slot < static_cast<int>(top_blocks.size()); ++slot) {
              if (cur.max_abs > top_blocks[slot].max_abs) {
                for (int j = static_cast<int>(top_blocks.size()) - 1; j > slot; --j) {
                  top_blocks[j] = top_blocks[j - 1];
                }
                top_blocks[slot] = cur;
                break;
              }
            }
          }
        }
        for (const auto& e : top_blocks) {
          if (e.row < 0) break;
          std::fprintf(stderr,
                       "[mxfp] c_block_err expert=%d row=%d cb=%d scale=%g max_abs=%g mean_abs=%g\n",
                       local_expert_idx, e.row, e.cb, e.scale, e.max_abs, e.mean_abs);
        }
      }
      const char* env_dump_c_block = std::getenv("FIB_MOE_TC_DUMP_C_BLOCK");
      if (env_dump_c_block != nullptr && env_dump_c_block[0] == '1' && n_rows > 0) {
        float c_ref_host[8] = {0};
        float c_cur_host[8] = {0};
        float c_deq_host[8] = {0};
        uint8_t q_host[8] = {0};
        float scale_host[4] = {0};
        cudaMemcpyAsync(c_ref_host, c_ref_dev, sizeof(c_ref_host), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(c_cur_host, c_dev_, sizeof(c_cur_host), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(c_deq_host, c_deq_dev, sizeof(c_deq_host), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(q_host, tc_c_fp8_dev_, sizeof(q_host), cudaMemcpyDeviceToHost, stream);
        int scale_cols = intermediate_blocks_;
        if (c_scale_gran_m == 128) {
          int row_blocks = (padded_rows + 127) / 128;
          int to_copy = std::min(scale_cols, 4);
          if (scale_major_k) {
            cudaMemcpyAsync(scale_host, tc_c_scale_dev_, to_copy * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
          } else {
            cudaMemcpyAsync(scale_host, tc_c_scale_dev_, to_copy * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
          }
          (void)row_blocks;
        } else {
          int to_copy = std::min(scale_cols, 4);
          if (scale_major_k) {
            cudaMemcpyAsync(scale_host, tc_c_scale_dev_, to_copy * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
          } else {
            cudaMemcpyAsync(scale_host, tc_c_scale_dev_, to_copy * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
          }
        }
        cudaStreamSynchronize(stream);
        std::fprintf(stderr,
                     "[mxfp] dump_c_block expert=%d scale_gran_m=%d scale_major_k=%d "
                     "scale0-3=[%g %g %g %g]\n",
                     local_expert_idx, c_scale_gran_m, scale_major_k ? 1 : 0, scale_host[0],
                     scale_host[1], scale_host[2], scale_host[3]);
        std::fprintf(stderr,
                     "[mxfp] dump_c_block ref0-7=[%g %g %g %g %g %g %g %g]\n",
                     c_ref_host[0], c_ref_host[1], c_ref_host[2], c_ref_host[3], c_ref_host[4],
                     c_ref_host[5], c_ref_host[6], c_ref_host[7]);
        std::fprintf(stderr,
                     "[mxfp] dump_c_block cur0-7=[%g %g %g %g %g %g %g %g]\n",
                     c_cur_host[0], c_cur_host[1], c_cur_host[2], c_cur_host[3], c_cur_host[4],
                     c_cur_host[5], c_cur_host[6], c_cur_host[7]);
        std::fprintf(stderr,
                     "[mxfp] dump_c_block q0-7=[%u %u %u %u %u %u %u %u]\n", q_host[0], q_host[1],
                     q_host[2], q_host[3], q_host[4], q_host[5], q_host[6], q_host[7]);
        std::fprintf(stderr,
                     "[mxfp] dump_c_block deq0-7=[%g %g %g %g %g %g %g %g]\n",
                     c_deq_host[0], c_deq_host[1], c_deq_host[2], c_deq_host[3], c_deq_host[4],
                     c_deq_host[5], c_deq_host[6], c_deq_host[7]);
      }
      std::fflush(stderr);
    }
  }
  int64_t w2_transpose_elems = static_cast<int64_t>(hidden_) * intermediate_;
  if (transpose_b) {
    transpose_rowmajor_nk_to_colmajor_nk_kernel<<<
        static_cast<int>((w2_transpose_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        w2_e, hidden_, intermediate_, tc_b_col_dev_);
  }
  FP8* w2_tc = transpose_b ? reinterpret_cast<FP8*>(tc_b_col_dev_)
                           : const_cast<FP8*>(reinterpret_cast<const FP8*>(w2_e));
  int w2_scale_elems = hidden_blocks_ * intermediate_blocks_;
  float* w2_scale_tc = const_cast<float*>(s2_e);
  if (scale_major_k) {
    (void)w2_scale_elems;
  } else {
    transpose_scale_nblock_kblock_to_kblock_nblock_kernel<<<
        (w2_scale_elems + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        s2_e, hidden_blocks_, intermediate_blocks_, tc_b_scale_dev_);
    w2_scale_tc = tc_b_scale_dev_;
  }

  auto run_gemm2_1sm = [&]() {
    TcGemmDispatch g2_dispatch = select_trtllm_like_gemm2_dispatch(n_rows, padded_rows);
    int g2_tile_m = select_tile_m_from_env("FIB_MOE_G2_TILE_M", g2_dispatch.tile_m);
    if (scale_major_k) {
      if (c_scale_gran_m == 128) {
        return launch_cutlass_blockscaled_group_gemm_sm100<128, true, 128, 1,
                                                           cutlass::float_e4m3_t,
                                                           cutlass::float_e4m3_t, float>(
            tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
            32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
            tc_c_scale_dev_, w2_scale_tc, tc_d_f32_dev_, padded_rows, hidden_,
            intermediate_, stream);
      }
      if (g2_tile_m == 64) {
        return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 64, 1,
                                                           cutlass::float_e4m3_t,
                                                           cutlass::float_e4m3_t, float>(
            tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
            32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
            tc_c_scale_dev_, w2_scale_tc, tc_d_f32_dev_,
            padded_rows, hidden_, intermediate_, stream);
      }
      return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 128, 1, cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
          tc_c_scale_dev_, w2_scale_tc, tc_d_f32_dev_,
          padded_rows, hidden_, intermediate_, stream);
    }
    if (c_scale_gran_m == 128) {
      return launch_cutlass_blockscaled_group_gemm_sm100<128, false, 128, 1,
                                                         cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
          tc_c_scale_dev_, w2_scale_tc, tc_d_f32_dev_,
          padded_rows, hidden_, intermediate_, stream);
    }
    if (g2_tile_m == 64) {
      return launch_cutlass_blockscaled_group_gemm_sm100<1, false, 64, 1, cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
          tc_c_scale_dev_, w2_scale_tc, tc_d_f32_dev_,
          padded_rows, hidden_, intermediate_, stream);
    }
    return launch_cutlass_blockscaled_group_gemm_sm100<1, false, 128, 1, cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
        tc_c_scale_dev_, w2_scale_tc, tc_d_f32_dev_,
        padded_rows, hidden_, intermediate_, stream);
  };
  auto run_gemm2_2sm = [&]() {
    if (scale_major_k) {
      if (c_scale_gran_m == 128) {
        return launch_cutlass_blockscaled_group_gemm_sm100<128, true, 256, 2,
                                                           cutlass::float_e4m3_t,
                                                           cutlass::float_e4m3_t, float>(
            tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
            32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
            tc_c_scale_dev_, w2_scale_tc, tc_d_f32_dev_, padded_rows, hidden_,
            intermediate_, stream);
      }
      return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 256, 2, cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
          tc_c_scale_dev_, w2_scale_tc, tc_d_f32_dev_,
          padded_rows, hidden_, intermediate_, stream);
    }
    if (c_scale_gran_m == 128) {
      return launch_cutlass_blockscaled_group_gemm_sm100<128, false, 256, 2,
                                                         cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
          tc_c_scale_dev_, w2_scale_tc, tc_d_f32_dev_,
          padded_rows, hidden_, intermediate_, stream);
    }
    return launch_cutlass_blockscaled_group_gemm_sm100<1, false, 256, 2, cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
        tc_c_scale_dev_, w2_scale_tc, tc_d_f32_dev_,
        padded_rows, hidden_, intermediate_, stream);
  };
  TcGemmDispatch g2_dispatch = select_trtllm_like_gemm2_dispatch(n_rows, padded_rows);
  const char* env_g2_2sm_min_rows = std::getenv("FIB_MOE_G2_2SM_MIN_ROWS");
  const char* env_g2_mma_sm = std::getenv("FIB_MOE_G2_MMA_SM");
  int g2_mma_sm = select_mma_sm_from_env_value(env_g2_mma_sm, g2_dispatch.mma_sms);
  if (env_g2_mma_sm == nullptr && env_g2_2sm_min_rows != nullptr) {
    g2_mma_sm = (padded_rows >= std::atoi(env_g2_2sm_min_rows)) ? 2 : 1;
  }
  cudaError_t st2 = (g2_mma_sm == 2) ? run_gemm2_2sm() : run_gemm2_1sm();
  if (st2 != cudaSuccess) return;

  int64_t scatter_elems = static_cast<int64_t>(n_rows) * hidden_;
  scatter_float_weighted_kernel<<<static_cast<int>((scatter_elems + kThreads - 1) / kThreads),
                                  kThreads, 0, stream>>>(
      tc_d_f32_dev_, hidden_, n_rows, permuted_tok_e, permuted_w_e, out_acc_dev);
#else
  return RunExpertPermuted(nullptr, t, n_rows, permuted_tok_e, permuted_w_e, local_expert_idx,
                           gemm1_w_dev, gemm1_s_dev, gemm2_w_dev, gemm2_s_dev, out_acc_dev,
                           stream);
#endif
}

}  // namespace mxfp
