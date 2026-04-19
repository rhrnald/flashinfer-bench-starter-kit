#include "mxfp_gemm_module.h"
#include "moe_tc_backend.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#if __has_include(<flashinfer/gemm/group_gemm_fp8_groupwise_sm100.cuh>)
#include <cuda_fp8.h>
#include <cutlass/numeric_types.h>
#include <flashinfer/gemm/group_gemm_fp8_groupwise_sm100.cuh>
#define FIB_HAS_FLASHINFER_FP8_GROUP_GEMM_SM100 1
#else
#define FIB_HAS_FLASHINFER_FP8_GROUP_GEMM_SM100 0
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace mxfp {

namespace {

inline float siluf_host(float x) { return x / (1.0f + std::exp(-x)); }

__device__ __forceinline__ float fp8_e4m3fn_to_float_device(uint8_t x) {
  int sign = (x & 0x80) ? -1 : 1;
  int exp = (x >> 3) & 0x0f;
  int mant = x & 0x07;

  if (exp == 0) {
    if (mant == 0) {
      return sign == 1 ? 0.0f : -0.0f;
    }
    float frac = static_cast<float>(mant) * 0.125f;
    return sign * ldexpf(frac, -6);
  }

  float frac = 1.0f + static_cast<float>(mant) * 0.125f;
  return sign * ldexpf(frac, exp - 7);
}

__device__ __forceinline__ float siluf_device(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float bf16_to_float_device(uint16_t bits) {
  uint32_t u32 = static_cast<uint32_t>(bits) << 16;
  return __uint_as_float(u32);
}

__device__ __forceinline__ float f16_to_float_device(uint16_t bits) {
  union {
    uint16_t u;
    __half h;
  } v;
  v.u = bits;
  return __half2float(v.h);
}

__device__ __forceinline__ uint8_t float_to_e4m3_device(float x) {
#if FIB_HAS_FLASHINFER_FP8_GROUP_GEMM_SM100
  __nv_fp8_e4m3 y(x);
  return *reinterpret_cast<uint8_t*>(&y);
#else
  (void)x;
  return 0;
#endif
}

// Temporary FP8-unit emulation: quantize intermediate values to an E4M3FN-like grid.
__device__ __forceinline__ float quantize_e4m3fn_like(float x) {
  if (!isfinite(x) || x == 0.0f) return 0.0f;
  float sign = x < 0.0f ? -1.0f : 1.0f;
  float ax = fabsf(x);
  const float kMax = 448.0f;
  const float kMinSub = ldexpf(1.0f, -9);  // 2^-9
  if (ax >= kMax) return sign * kMax;
  if (ax < kMinSub) return 0.0f;

  int e2;
  float m = frexpf(ax, &e2);  // ax = m * 2^e2, m in [0.5,1)
  int e = e2 - 1;             // normalized exponent in [-6, 8]

  if (e < -6) {
    float q = nearbyintf(ax / kMinSub);
    q = fminf(7.0f, fmaxf(0.0f, q));
    return sign * q * kMinSub;
  }
  if (e > 8) return sign * kMax;

  float base = ldexpf(1.0f, e);
  float mf = ax / base;                         // [1,2)
  float qm = nearbyintf((mf - 1.0f) * 8.0f);   // 3-bit frac
  qm = fminf(7.0f, fmaxf(0.0f, qm));
  float qmf = 1.0f + qm * 0.125f;
  return sign * qmf * base;
}

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

__global__ void swiglu_quantize_bf16_to_fp8_kernel(const uint16_t* __restrict__ g1_bf16,
                                                   int intermediate, int n_rows, int padded_rows,
                                                   uint8_t* __restrict__ c_fp8,
                                                   float* __restrict__ c_scale) {
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
    c_scale[static_cast<int64_t>(ib) * padded_rows + row] = scale;
  }
  if (tid < 128 && i < intermediate) {
    c_fp8[static_cast<int64_t>(row) * intermediate + i] =
        (row < n_rows) ? float_to_e4m3_device(v / scale) : 0;
  }
}

__global__ void swiglu_quantize_f16_to_fp8_kernel(const uint16_t* __restrict__ g1_f16,
                                                  int intermediate, int n_rows, int padded_rows,
                                                  uint8_t* __restrict__ c_fp8,
                                                  float* __restrict__ c_scale) {
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
    c_scale[static_cast<int64_t>(ib) * padded_rows + row] = scale;
  }
  if (tid < 128 && i < intermediate) {
    c_fp8[static_cast<int64_t>(row) * intermediate + i] =
        (row < n_rows) ? float_to_e4m3_device(v / scale) : 0;
  }
}

__global__ void scatter_bf16_weighted_kernel(const uint16_t* __restrict__ d_bf16, int hidden,
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
  float v = bf16_to_float_device(d_bf16[idx]);
  out_acc[static_cast<int64_t>(tok) * hidden + h] += w * v;
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

}  // namespace

float fp8_e4m3fn_to_float(uint8_t x) {
  int sign = (x & 0x80) ? -1 : 1;
  int exp = (x >> 3) & 0x0f;
  int mant = x & 0x07;

  if (exp == 0) {
    if (mant == 0) {
      return sign == 1 ? 0.0f : -0.0f;
    }
    float frac = static_cast<float>(mant) / 8.0f;
    return sign * std::ldexp(frac, -6);
  }

  float frac = 1.0f + static_cast<float>(mant) / 8.0f;
  return sign * std::ldexp(frac, exp - 7);
}

HostMxfpGemmModule::HostMxfpGemmModule(int hidden, int intermediate, int block)
    : hidden_(hidden),
      intermediate_(intermediate),
      block_(block),
      gemm1_out_(2 * intermediate),
      hidden_blocks_(hidden / block),
      intermediate_blocks_(intermediate / block),
      gemm1_out_blocks_(gemm1_out_ / block),
      w13_fp8_(gemm1_weight_elems()),
      w13_scale_(gemm1_scale_elems()),
      w2_fp8_(gemm2_weight_elems()),
      w2_scale_(gemm2_scale_elems()) {}

size_t HostMxfpGemmModule::gemm1_weight_elems() const {
  return static_cast<size_t>(gemm1_out_) * static_cast<size_t>(hidden_);
}

size_t HostMxfpGemmModule::gemm1_scale_elems() const {
  return static_cast<size_t>(gemm1_out_blocks_) * static_cast<size_t>(hidden_blocks_);
}

size_t HostMxfpGemmModule::gemm2_weight_elems() const {
  return static_cast<size_t>(hidden_) * static_cast<size_t>(intermediate_);
}

size_t HostMxfpGemmModule::gemm2_scale_elems() const {
  return static_cast<size_t>(hidden_blocks_) * static_cast<size_t>(intermediate_blocks_);
}

void HostMxfpGemmModule::load_expert_from_device(int local_expert_idx, const uint8_t* gemm1_weights_dev,
                                                  const float* gemm1_scales_dev,
                                                  const uint8_t* gemm2_weights_dev,
                                                  const float* gemm2_scales_dev,
                                                  cudaStream_t stream) {
  size_t w13_elems = gemm1_weight_elems();
  size_t w13s_elems = gemm1_scale_elems();
  size_t w2_elems = gemm2_weight_elems();
  size_t w2s_elems = gemm2_scale_elems();

  size_t w13_off = static_cast<size_t>(local_expert_idx) * w13_elems;
  size_t w13s_off = static_cast<size_t>(local_expert_idx) * w13s_elems;
  size_t w2_off = static_cast<size_t>(local_expert_idx) * w2_elems;
  size_t w2s_off = static_cast<size_t>(local_expert_idx) * w2s_elems;

  cudaMemcpyAsync(w13_fp8_.data(), gemm1_weights_dev + w13_off, w13_elems, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(w13_scale_.data(), gemm1_scales_dev + w13s_off, w13s_elems * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(w2_fp8_.data(), gemm2_weights_dev + w2_off, w2_elems, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(w2_scale_.data(), gemm2_scales_dev + w2s_off, w2s_elems * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
}

void HostMxfpGemmModule::gemm1_matvec(const float* a_row, float* g1_out) const {
  for (int j = 0; j < gemm1_out_; ++j) {
    int jb = j / block_;
    float acc = 0.0f;
    const uint8_t* w13_row = w13_fp8_.data() + static_cast<size_t>(j) * hidden_;
    for (int h = 0; h < hidden_; ++h) {
      int hb = h / block_;
      float s = w13_scale_[static_cast<size_t>(jb) * hidden_blocks_ + static_cast<size_t>(hb)];
      float wv = fp8_e4m3fn_to_float(w13_row[h]) * s;
      acc += a_row[h] * wv;
    }
    g1_out[j] = acc;
  }
}

void HostMxfpGemmModule::swiglu(const float* g1, int intermediate, float* c_out) {
  for (int i = 0; i < intermediate; ++i) {
    float x1 = g1[i];
    float x2 = g1[i + intermediate];
    c_out[i] = x1 * siluf_host(x2);
  }
}

void HostMxfpGemmModule::gemm2_matvec_accumulate(const float* c, float weight, float* out_row) const {
  for (int h = 0; h < hidden_; ++h) {
    int hb = h / block_;
    float acc = 0.0f;
    const uint8_t* w2_row = w2_fp8_.data() + static_cast<size_t>(h) * intermediate_;
    for (int i = 0; i < intermediate_; ++i) {
      int ib = i / block_;
      float s = w2_scale_[static_cast<size_t>(hb) * intermediate_blocks_ + static_cast<size_t>(ib)];
      float wv = fp8_e4m3fn_to_float(w2_row[i]) * s;
      acc += c[i] * wv;
    }
    out_row[h] += weight * acc;
  }
}

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
      tc5090_env_(false),
      tc5090_backend_(nullptr),
      g1_dev_(nullptr),
      c_dev_(nullptr),
      tc_max_rows_(0),
      tc_path_env_(false),
      tc_a_fp8_dev_(nullptr),
      tc_b_col_dev_(nullptr),
      tc_a_scale_dev_(nullptr),
      tc_b_scale_dev_(nullptr),
      tc_g1_bf16_dev_(nullptr),
      tc_c_fp8_dev_(nullptr),
      tc_c_scale_dev_(nullptr),
      tc_d_bf16_dev_(nullptr),
      tc_m_indptr_dev_(nullptr),
      tc_int_workspace_dev_(nullptr),
      tc_float_workspace_dev_(nullptr) {
  const char* env = std::getenv("FIB_EMULATE_FP8_UNIT");
  emulate_fp8_unit_ = (env != nullptr && env[0] == '1');
  const char* env_fp16_op = std::getenv("FIB_EMULATE_FP16_OPERANDS");
  emulate_fp16_operands_ = (env_fp16_op != nullptr && env_fp16_op[0] == '1');
  const char* env_acc = std::getenv("FIB_EMULATE_FP8_ACC_HALF");
  emulate_acc_half_ = (env_acc != nullptr && env_acc[0] == '1');
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
  const char* env_tc5090 = std::getenv("FIB_MOE_TC5090");
  tc5090_env_ = (env_tc5090 != nullptr && env_tc5090[0] == '1');
  if (tc5090_env_) {
    MoeTcBackendConfig cfg = {
        hidden_, intermediate_, block_, hidden_blocks_, intermediate_blocks_, gemm1_out_blocks_};
    tc5090_backend_ = CreateMoeTcBackend5090Temp(cfg);
    if (tc5090_backend_ != nullptr && tc5090_backend_->IsAvailable()) {
      std::fprintf(stderr, "[mxfp] FIB_MOE_TC5090=1 using backend=%s\n",
                   tc5090_backend_->BackendName());
    } else {
      std::fprintf(stderr,
                   "[mxfp] FIB_MOE_TC5090=1 requested but backend unavailable; falling back\n");
      tc5090_backend_.reset();
    }
  }
  const char* env_tc = std::getenv("FIB_MOE_TC");
  tc_path_env_ = (env_tc != nullptr && env_tc[0] == '1');
  if (tc_path_env_) {
    std::fprintf(stderr, "[mxfp] FIB_MOE_TC=1 requested; FlashInfer/CUTLASS SM100 path %s\n",
                 SupportsTcPath() ? "available" : "not available at compile time");
  }
  if (emulate_fp8_unit_ || emulate_fp16_operands_ || emulate_acc_half_ || tc5090_env_ ||
      tc_path_env_) {
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
  if (tc_g1_bf16_dev_ != nullptr) cudaFree(tc_g1_bf16_dev_);
  if (tc_c_fp8_dev_ != nullptr) cudaFree(tc_c_fp8_dev_);
  if (tc_c_scale_dev_ != nullptr) cudaFree(tc_c_scale_dev_);
  if (tc_d_bf16_dev_ != nullptr) cudaFree(tc_d_bf16_dev_);
  if (tc_m_indptr_dev_ != nullptr) cudaFree(tc_m_indptr_dev_);
  if (tc_int_workspace_dev_ != nullptr) cudaFree(tc_int_workspace_dev_);
  if (tc_float_workspace_dev_ != nullptr) cudaFree(tc_float_workspace_dev_);
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
  return tc_path_env_ && FIB_HAS_FLASHINFER_FP8_GROUP_GEMM_SM100;
}

void DeviceMxfpGemmModule::EnsureTcWorkspace(int rows) {
  int padded_rows = (rows + 3) & ~3;
  if (padded_rows <= tc_max_rows_ && tc_a_fp8_dev_ != nullptr) return;

  if (tc_a_fp8_dev_ != nullptr) cudaFree(tc_a_fp8_dev_);
  if (tc_b_col_dev_ != nullptr) cudaFree(tc_b_col_dev_);
  if (tc_a_scale_dev_ != nullptr) cudaFree(tc_a_scale_dev_);
  if (tc_b_scale_dev_ != nullptr) cudaFree(tc_b_scale_dev_);
  if (tc_g1_bf16_dev_ != nullptr) cudaFree(tc_g1_bf16_dev_);
  if (tc_c_fp8_dev_ != nullptr) cudaFree(tc_c_fp8_dev_);
  if (tc_c_scale_dev_ != nullptr) cudaFree(tc_c_scale_dev_);
  if (tc_d_bf16_dev_ != nullptr) cudaFree(tc_d_bf16_dev_);
  if (tc_m_indptr_dev_ != nullptr) cudaFree(tc_m_indptr_dev_);
  if (tc_int_workspace_dev_ != nullptr) cudaFree(tc_int_workspace_dev_);
  if (tc_float_workspace_dev_ != nullptr) cudaFree(tc_float_workspace_dev_);

  tc_max_rows_ = padded_rows;
  constexpr size_t kCutlassWorkspaceBytes = 32ull * 1024ull * 1024ull;
  cudaError_t e = cudaSuccess;
  e = cudaMalloc(&tc_a_fp8_dev_, static_cast<size_t>(padded_rows) * hidden_ * sizeof(uint8_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_a_fp8_dev_");
  e = cudaMalloc(&tc_b_col_dev_, static_cast<size_t>(gemm1_out_) * hidden_ * sizeof(uint8_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_b_col_dev_");
  e = cudaMalloc(&tc_a_scale_dev_, static_cast<size_t>(padded_rows) * hidden_blocks_ * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_a_scale_dev_");
  e = cudaMalloc(&tc_b_scale_dev_, static_cast<size_t>(gemm1_out_blocks_) * hidden_blocks_ * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_b_scale_dev_");
  e = cudaMalloc(&tc_g1_bf16_dev_, static_cast<size_t>(padded_rows) * gemm1_out_ * sizeof(uint16_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_g1_bf16_dev_");
  e = cudaMalloc(&tc_c_fp8_dev_, static_cast<size_t>(padded_rows) * intermediate_ * sizeof(uint8_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_c_fp8_dev_");
  e = cudaMalloc(&tc_c_scale_dev_, static_cast<size_t>(padded_rows) * intermediate_blocks_ * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_c_scale_dev_");
  e = cudaMalloc(&tc_d_bf16_dev_, static_cast<size_t>(padded_rows) * hidden_ * sizeof(uint16_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_d_bf16_dev_");
  e = cudaMalloc(&tc_m_indptr_dev_, 2 * sizeof(int));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_m_indptr_dev_");
  e = cudaMalloc(&tc_int_workspace_dev_, kCutlassWorkspaceBytes);
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_int_workspace_dev_");
  e = cudaMalloc(&tc_float_workspace_dev_, kCutlassWorkspaceBytes);
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_float_workspace_dev_");
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
  // n_rows == T_e (tokens routed to this expert on local rank).
  // If T_e == 0, skip all expert work.
  if (n_rows <= 0) return;

  size_t w13_elems = static_cast<size_t>(gemm1_out_) * static_cast<size_t>(hidden_);
  size_t w13s_elems = static_cast<size_t>(gemm1_out_blocks_) * static_cast<size_t>(hidden_blocks_);
  size_t w2_elems = static_cast<size_t>(hidden_) * static_cast<size_t>(intermediate_);
  size_t w2s_elems = static_cast<size_t>(hidden_blocks_) * static_cast<size_t>(intermediate_blocks_);

  const uint8_t* w13_e = gemm1_w_dev + static_cast<size_t>(local_expert_idx) * w13_elems;
  const float* s13_e = gemm1_s_dev + static_cast<size_t>(local_expert_idx) * w13s_elems;
  const uint8_t* w2_e = gemm2_w_dev + static_cast<size_t>(local_expert_idx) * w2_elems;
  const float* s2_e = gemm2_s_dev + static_cast<size_t>(local_expert_idx) * w2s_elems;

  if (tc5090_backend_ != nullptr) {
    cudaError_t st1 = tc5090_backend_->RunStep1Fused(a_dev, n_rows, permuted_tok_e, w13_e, s13_e,
                                                     c_dev_, stream);
    if (st1 == cudaSuccess) {
      cudaError_t st2 = tc5090_backend_->RunStep2(c_dev_, n_rows, permuted_tok_e, permuted_w_e,
                                                  w2_e, s2_e, out_acc_dev, stream);
      if (st2 == cudaSuccess) {
        return;
      }
      std::fprintf(stderr,
                   "[mxfp] 5090 temp backend step2 launch failed (%d), fallback to legacy permuted path\n",
                   static_cast<int>(st2));
    } else {
      std::fprintf(stderr,
                   "[mxfp] 5090 temp backend step1 launch failed (%d), fallback to legacy permuted path\n",
                   static_cast<int>(st1));
    }
    cudaGetLastError();
  }

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

void DeviceMxfpGemmModule::RunExpertPermutedTc(const uint8_t* hidden_fp8_dev,
                                               const float* hidden_scale_dev, int64_t t,
                                               int n_rows, const int* permuted_tok_e,
                                               const float* permuted_w_e, int local_expert_idx,
                                               const uint8_t* gemm1_w_dev,
                                               const float* gemm1_s_dev,
                                               const uint8_t* gemm2_w_dev,
                                               const float* gemm2_s_dev,
                                               float* out_acc_dev, cudaStream_t stream) {
  // n_rows == T_e (tokens routed to this expert on local rank).
  if (n_rows <= 0) return;
  if (!SupportsTcPath()) {
    return RunExpertPermuted(nullptr, t, n_rows, permuted_tok_e, permuted_w_e, local_expert_idx,
                             gemm1_w_dev, gemm1_s_dev, gemm2_w_dev, gemm2_s_dev, out_acc_dev,
                             stream);
  }

#if FIB_HAS_FLASHINFER_FP8_GROUP_GEMM_SM100
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
  // The B200 contract probe matches unchanged [N,K] physical storage for this
  // FlashInfer groupwise GEMM path.
  const bool transpose_b = std::getenv("FIB_MOE_TC_TRANSPOSE_B") != nullptr;

  using FP8 = cutlass::float_e4m3_t;
  using F16 = cutlass::half_t;
  using BF16 = cutlass::bfloat16_t;

  constexpr int kThreads = 256;
  int64_t a_elems = static_cast<int64_t>(padded_rows) * hidden_;
  int64_t a_scale_elems = static_cast<int64_t>(padded_rows) * hidden_blocks_;
  gather_hidden_fp8_rows_kernel<<<static_cast<int>((a_elems + kThreads - 1) / kThreads),
                                  kThreads, 0, stream>>>(
      hidden_fp8_dev, permuted_tok_e, n_rows, padded_rows, hidden_, tc_a_fp8_dev_);
  gather_hidden_scale_rows_kernel<<<static_cast<int>((a_scale_elems + kThreads - 1) / kThreads),
                                    kThreads, 0, stream>>>(
      hidden_scale_dev, permuted_tok_e, t, n_rows, padded_rows, hidden_blocks_, tc_a_scale_dev_);
  write_single_group_indptr_kernel<<<1, 1, 0, stream>>>(padded_rows, tc_m_indptr_dev_);
  int64_t w13_transpose_elems = static_cast<int64_t>(gemm1_out_) * hidden_;
  if (transpose_b) {
    transpose_rowmajor_nk_to_colmajor_nk_kernel<<<
        static_cast<int>((w13_transpose_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        w13_e, gemm1_out_, hidden_, tc_b_col_dev_);
  }
  FP8* w13_tc = transpose_b ? reinterpret_cast<FP8*>(tc_b_col_dev_)
                            : const_cast<FP8*>(reinterpret_cast<const FP8*>(w13_e));
  // Use the MN-major scale contract validated by scripts/probe_gemm1_contract.py:
  // A scale [Kblk,M], B scale [Kblk,Nblk], raw [N,K] B storage.
  int w13_scale_elems = gemm1_out_blocks_ * hidden_blocks_;
  transpose_scale_nblock_kblock_to_kblock_nblock_kernel<<<
      (w13_scale_elems + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      s13_e, gemm1_out_blocks_, hidden_blocks_, tc_b_scale_dev_);

  auto run_gemm1_1sm = [&]() {
    return flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM100<
        1, 128, 128, false, 1>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
        w13_tc, tc_a_scale_dev_,
        tc_b_scale_dev_, reinterpret_cast<F16*>(tc_g1_bf16_dev_), tc_m_indptr_dev_,
        padded_rows, gemm1_out_, hidden_, 1, stream);
  };
  auto run_gemm1_2sm = [&]() {
    return flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM100<
        1, 128, 128, false, 2>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_),
        w13_tc, tc_a_scale_dev_,
        tc_b_scale_dev_, reinterpret_cast<F16*>(tc_g1_bf16_dev_), tc_m_indptr_dev_,
        padded_rows, gemm1_out_, hidden_, 1, stream);
  };
  cudaError_t st1 = (padded_rows >= 256) ? run_gemm1_2sm() : run_gemm1_1sm();
  if (st1 != cudaSuccess) return;

  if (std::getenv("FIB_MOE_TC_GEMM1_ONLY") != nullptr) {
    EnsureWorkspace(n_rows, stream);
    int64_t g1_elems = static_cast<int64_t>(n_rows) * gemm1_out_;
    f16_matrix_to_float_kernel<<<static_cast<int>((g1_elems + kThreads - 1) / kThreads),
                                 kThreads, 0, stream>>>(
        tc_g1_bf16_dev_, g1_elems, g1_dev_);
    int64_t c_elems = static_cast<int64_t>(n_rows) * intermediate_;
    int64_t out_elems = static_cast<int64_t>(n_rows) * hidden_;
    swiglu_permuted_kernel<<<static_cast<int>((c_elems + kThreads - 1) / kThreads),
                              kThreads, 0, stream>>>(
        g1_dev_, intermediate_, n_rows, false, c_dev_);
    gemm2_scatter_accumulate_kernel<<<static_cast<int>((out_elems + kThreads - 1) / kThreads),
                                      kThreads, 0, stream>>>(
        c_dev_, hidden_, intermediate_, block_, intermediate_blocks_, n_rows, permuted_tok_e,
        permuted_w_e, w2_e, s2_e, emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_,
        out_acc_dev);
    return;
  }

  dim3 qgrid(padded_rows, intermediate_blocks_);
  swiglu_quantize_f16_to_fp8_kernel<<<qgrid, 128, 128 * sizeof(float), stream>>>(
      tc_g1_bf16_dev_, intermediate_, n_rows, padded_rows, tc_c_fp8_dev_, tc_c_scale_dev_);
  int64_t w2_transpose_elems = static_cast<int64_t>(hidden_) * intermediate_;
  if (transpose_b) {
    transpose_rowmajor_nk_to_colmajor_nk_kernel<<<
        static_cast<int>((w2_transpose_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        w2_e, hidden_, intermediate_, tc_b_col_dev_);
  }
  FP8* w2_tc = transpose_b ? reinterpret_cast<FP8*>(tc_b_col_dev_)
                           : const_cast<FP8*>(reinterpret_cast<const FP8*>(w2_e));
  int w2_scale_elems = hidden_blocks_ * intermediate_blocks_;
  transpose_scale_nblock_kblock_to_kblock_nblock_kernel<<<
      (w2_scale_elems + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      s2_e, hidden_blocks_, intermediate_blocks_, tc_b_scale_dev_);

  auto run_gemm2_1sm = [&]() {
    return flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM100<
        1, 128, 128, false, 1>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_),
        w2_tc, tc_c_scale_dev_,
        tc_b_scale_dev_, reinterpret_cast<BF16*>(tc_d_bf16_dev_), tc_m_indptr_dev_,
        padded_rows, hidden_, intermediate_, 1, stream);
  };
  auto run_gemm2_2sm = [&]() {
    return flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM100<
        1, 128, 128, false, 2>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_),
        w2_tc, tc_c_scale_dev_,
        tc_b_scale_dev_, reinterpret_cast<BF16*>(tc_d_bf16_dev_), tc_m_indptr_dev_,
        padded_rows, hidden_, intermediate_, 1, stream);
  };
  cudaError_t st2 = (padded_rows >= 256) ? run_gemm2_2sm() : run_gemm2_1sm();
  if (st2 != cudaSuccess) return;

  int64_t scatter_elems = static_cast<int64_t>(n_rows) * hidden_;
  scatter_bf16_weighted_kernel<<<static_cast<int>((scatter_elems + kThreads - 1) / kThreads),
                                 kThreads, 0, stream>>>(
      tc_d_bf16_dev_, hidden_, n_rows, permuted_tok_e, permuted_w_e, out_acc_dev);
#else
  return RunExpertPermuted(nullptr, t, n_rows, permuted_tok_e, permuted_w_e, local_expert_idx,
                           gemm1_w_dev, gemm1_s_dev, gemm2_w_dev, gemm2_s_dev, out_acc_dev,
                           stream);
#endif
}

}  // namespace mxfp
