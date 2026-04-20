#include "backend_a100.h"
#include "moe_tc_backend_b200_step1_direct.cuh"
#include "moe_tc_backend_b200_step2_direct.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace mxfp {

namespace {

inline bool IsSm100Device() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return false;
  int major = 0;
  int minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) return false;
  if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) return false;
  return major == 10 && minor == 0;
}

inline bool HasAnyCudaDevice() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
  return n > 0;
}

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

// Approximate positive E8M0 scale quantization with nearest power-of-two.
__device__ __forceinline__ float quantize_scale_e8m0_like_device(float x) {
  if (!isfinite(x) || x == 0.0f) return 0.0f;
  float sign = x < 0.0f ? -1.0f : 1.0f;
  float ax = fabsf(x);
  int e2 = 0;
  float m = frexpf(ax, &e2);  // ax = m * 2^e2, m in [0.5, 1.0)
  int qexp = e2 - 1 + (m >= 0.70710678118f ? 1 : 0);
  return sign * ldexpf(1.0f, qexp);
}

__global__ void gemm1_kernel(const float* __restrict__ a, int64_t t, int hidden, int gemm1_out,
                             int block, int hidden_blocks, int local_expert_idx,
                             const float* __restrict__ local_weight, const uint8_t* __restrict__ w13,
                             const float* __restrict__ s13, bool emulate_fp8_unit,
                             bool emulate_fp16_operands, bool emulate_acc_half,
                             bool quantize_scale_e8m0,
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
    if (quantize_scale_e8m0) scale = quantize_scale_e8m0_like_device(scale);
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

// Permuted-path GEMM1. Input activation stays FP8+block-scale and is decoded
// on-the-fly inside the GEMM loop (no outer dequant pass).
__global__ void gemm1_permuted_kernel(const uint8_t* __restrict__ hidden_fp8,
                                      const float* __restrict__ hidden_scale,
                                      int64_t t, int hidden, int gemm1_out,
                                      int block, int hidden_blocks, int n_rows,
                                      const int* __restrict__ permuted_tok,
                                      const uint8_t* __restrict__ w13,
                                      const float* __restrict__ s13, bool emulate_fp8_unit,
                                      bool emulate_fp16_operands, bool emulate_acc_half,
                                      bool quantize_scale_e8m0,
                                      float* __restrict__ g1_perm) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * gemm1_out;
  if (idx >= total) return;

  int pr = static_cast<int>(idx / gemm1_out);
  int j = static_cast<int>(idx - static_cast<int64_t>(pr) * gemm1_out);
  int tok = permuted_tok[pr];

  int jb = j / block;
  const uint8_t* a_row = hidden_fp8 + static_cast<int64_t>(tok) * hidden;
  const uint8_t* w_row = w13 + static_cast<int64_t>(j) * hidden;
  float acc = 0.0f;
  __half acc_h = __float2half(0.0f);
  for (int hb = 0; hb < hidden_blocks; ++hb) {
    float a_scale = hidden_scale[static_cast<int64_t>(hb) * t + tok];
    float scale = s13[jb * hidden_blocks + hb];
    if (quantize_scale_e8m0) scale = quantize_scale_e8m0_like_device(scale);
    float block_raw = 0.0f;
    int h0 = hb * block;
    for (int u = 0; u < block; ++u) {
      int h = h0 + u;
      float wv_raw = fp8_e4m3fn_to_float_device(w_row[h]);
      float av = fp8_e4m3fn_to_float_device(a_row[h]) * a_scale;
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
                                                bool quantize_scale_e8m0,
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
    if (quantize_scale_e8m0) scale = quantize_scale_e8m0_like_device(scale);
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
                                 bool quantize_scale_e8m0,
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
    if (quantize_scale_e8m0) scale = quantize_scale_e8m0_like_device(scale);
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
      quantize_scale_e8m0_(false),
      b200_direct_enabled_(false),
      step1_hidden_tmap_dev_(nullptr),
      step1_w13_tmap_dev_(nullptr),
      g1_dev_(nullptr),
      c_dev_(nullptr) {
  // Submission-friendly policy:
  // - Direct B200 path is enabled by default.
  // - Set FIB_MOE_TCB200=0 to force-disable it (for local debugging).
  // - Set FIB_MOE_TCB200_STRICT=1 to fail fast if direct path is unavailable.
  const char* env_b200 = std::getenv("FIB_MOE_TCB200");
  const bool disable_b200 = (env_b200 != nullptr && env_b200[0] == '0');
  const bool strict_b200 = (std::getenv("FIB_MOE_TCB200_STRICT") != nullptr);
  if (!disable_b200) {
    const bool shape_ok = (hidden_ == 7168 && intermediate_ == 2048 && block_ == 128);
    if (!shape_ok) {
      std::fprintf(stderr,
                   "[mxfp] direct B200 disabled by shape mismatch (got H=%d I=%d B=%d, expected H=%d I=%d B=%d); using legacy grouped kernels\n",
                   hidden_, intermediate_, block_, 7168, 2048, 128);
    } else if (!HasAnyCudaDevice()) {
      std::fprintf(stderr, "[mxfp] direct B200 unavailable: no CUDA device visible; using legacy grouped kernels\n");
    } else if (!IsSm100Device()) {
      std::fprintf(stderr, "[mxfp] direct B200 unavailable: device is not SM100; using legacy grouped kernels\n");
    } else {
      b200_direct_enabled_ = true;
      std::fprintf(stderr, "[mxfp] using direct B200 step kernels\n");
    }
  } else {
    std::fprintf(stderr, "[mxfp] FIB_MOE_TCB200=0 -> forcing legacy grouped kernels\n");
  }
  if (strict_b200 && !b200_direct_enabled_) {
    throw std::runtime_error(
        "FIB_MOE_TCB200_STRICT=1 but direct B200 step kernels are unavailable");
  }
  std::fflush(stderr);
}

DeviceMxfpGemmModule::~DeviceMxfpGemmModule() {
  if (step1_hidden_tmap_dev_ != nullptr) cudaFree(step1_hidden_tmap_dev_);
  if (step1_w13_tmap_dev_ != nullptr) cudaFree(step1_w13_tmap_dev_);
  if (g1_dev_ != nullptr) cudaFree(g1_dev_);
  if (c_dev_ != nullptr) cudaFree(c_dev_);
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
                                            emulate_fp8_unit_, emulate_fp16_operands_,
                                            emulate_acc_half_, quantize_scale_e8m0_, g1_dev_);
  swiglu_kernel<<<b2, kThreads, 0, stream>>>(g1_dev_, t, intermediate_, local_expert_idx,
                                             local_weight_dev, emulate_fp8_unit_, c_dev_);
  gemm2_acc_kernel<<<b3, kThreads, 0, stream>>>(c_dev_, t, hidden_, intermediate_, block_,
                                                intermediate_blocks_, local_expert_idx,
                                                local_weight_dev, w2_e, s2_e, emulate_fp8_unit_,
                                                emulate_fp16_operands_,
                                                emulate_acc_half_, quantize_scale_e8m0_,
                                                out_acc_dev);
}

void DeviceMxfpGemmModule::RunExpertPermuted(const uint8_t* hidden_fp8_dev,
                                             const float* hidden_scale_dev, int64_t t, int n_rows,
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

  constexpr int kThreads = 128;
  int64_t n_g1 = static_cast<int64_t>(n_rows) * gemm1_out_;
  int64_t n_c = static_cast<int64_t>(n_rows) * intermediate_;
  int64_t n_out = static_cast<int64_t>(n_rows) * hidden_;

  int b1 = static_cast<int>((n_g1 + kThreads - 1) / kThreads);
  int b2 = static_cast<int>((n_c + kThreads - 1) / kThreads);
  int b3 = static_cast<int>((n_out + kThreads - 1) / kThreads);

  gemm1_permuted_kernel<<<b1, kThreads, 0, stream>>>(
      hidden_fp8_dev, hidden_scale_dev, t, hidden_, gemm1_out_, block_, hidden_blocks_, n_rows,
      permuted_tok_e, w13_e, s13_e, emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_,
      quantize_scale_e8m0_, g1_dev_);
  swiglu_permuted_kernel<<<b2, kThreads, 0, stream>>>(g1_dev_, intermediate_, n_rows,
                                                      emulate_fp8_unit_, c_dev_);
  gemm2_scatter_accumulate_kernel<<<b3, kThreads, 0, stream>>>(
      c_dev_, hidden_, intermediate_, block_, intermediate_blocks_, n_rows, permuted_tok_e,
      permuted_w_e, w2_e, s2_e, emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_,
      quantize_scale_e8m0_, out_acc_dev);
}

cudaError_t DeviceMxfpGemmModule::RunStep1AllExpertsDirect(const uint8_t* hidden_fp8_dev,
                                                           const float* hidden_scale_dev, int64_t t,
                                                           const int* expert_t_valid,
                                                           const int* expert_offset,
                                                           const int* valid_token_idx,
                                                           const uint8_t* gemm1_w_dev,
                                                           const float* gemm1_s_dev,
                                                           float* c_perm_all_dev,
                                                           cudaStream_t stream) const {
  if (!b200_direct_enabled_) return cudaErrorNotSupported;
  if (step1_hidden_tmap_dev_ == nullptr) {
    if (cudaMalloc(&step1_hidden_tmap_dev_, sizeof(CUtensorMap)) != cudaSuccess) {
      step1_hidden_tmap_dev_ = nullptr;
    }
  }
  if (step1_w13_tmap_dev_ == nullptr) {
    if (cudaMalloc(&step1_w13_tmap_dev_, sizeof(CUtensorMap)) != cudaSuccess) {
      step1_w13_tmap_dev_ = nullptr;
    }
  }

  const CUtensorMap* hidden_tmap_dev = nullptr;
  const CUtensorMap* w13_tmap_dev = nullptr;
  if (step1_hidden_tmap_dev_ != nullptr && step1_w13_tmap_dev_ != nullptr) {
    CUtensorMap hidden_tmap_host{};
    CUtensorMap w13_tmap_host{};

    CUresult r_hidden = b200::ptx::EncodeTensorMap2D(
        &hidden_tmap_host, CU_TENSOR_MAP_DATA_TYPE_UINT8, const_cast<uint8_t*>(hidden_fp8_dev),
        static_cast<uint64_t>(t), static_cast<uint64_t>(hidden_),
        static_cast<uint64_t>(hidden_) * sizeof(uint8_t), 1u, 128u);
    CUresult r_w13 = b200::ptx::EncodeTensorMap2D(
        &w13_tmap_host, CU_TENSOR_MAP_DATA_TYPE_UINT8, const_cast<uint8_t*>(gemm1_w_dev),
        static_cast<uint64_t>(2 * intermediate_), static_cast<uint64_t>(hidden_),
        static_cast<uint64_t>(hidden_) * sizeof(uint8_t), 128u, 128u);

    if (r_hidden == CUDA_SUCCESS && r_w13 == CUDA_SUCCESS) {
      cudaError_t c1 = cudaMemcpyAsync(step1_hidden_tmap_dev_, &hidden_tmap_host, sizeof(CUtensorMap),
                                       cudaMemcpyHostToDevice, stream);
      cudaError_t c2 = cudaMemcpyAsync(step1_w13_tmap_dev_, &w13_tmap_host, sizeof(CUtensorMap),
                                       cudaMemcpyHostToDevice, stream);
      if (c1 == cudaSuccess && c2 == cudaSuccess) {
        hidden_tmap_dev = step1_hidden_tmap_dev_;
        w13_tmap_dev = step1_w13_tmap_dev_;
      }
    }
  }

  return b200::direct::LaunchStep1DirectAllExperts(hidden_fp8_dev, hidden_scale_dev, t,
                                                   expert_t_valid, expert_offset, valid_token_idx,
                                                   gemm1_w_dev, gemm1_s_dev, hidden_tmap_dev,
                                                   w13_tmap_dev, c_perm_all_dev, stream);
}

void DeviceMxfpGemmModule::RunStep2PermutedOnly(const float* c_perm_e, int n_rows,
                                                const int* permuted_tok_e,
                                                const float* permuted_w_e, int local_expert_idx,
                                                const uint8_t* gemm2_w_dev,
                                                const float* gemm2_s_dev, float* out_acc_dev,
                                                cudaStream_t stream) const {
  if (n_rows <= 0) return;

  size_t w2_elems = static_cast<size_t>(hidden_) * static_cast<size_t>(intermediate_);
  size_t w2s_elems = static_cast<size_t>(hidden_blocks_) * static_cast<size_t>(intermediate_blocks_);
  const uint8_t* w2_e = gemm2_w_dev + static_cast<size_t>(local_expert_idx) * w2_elems;
  const float* s2_e = gemm2_s_dev + static_cast<size_t>(local_expert_idx) * w2s_elems;

  if (b200_direct_enabled_) {
    cudaError_t st2 = b200::direct::LaunchStep2Direct(c_perm_e, n_rows, permuted_tok_e, permuted_w_e,
                                                      w2_e, s2_e, out_acc_dev, stream);
    if (st2 == cudaSuccess) return;
    std::fprintf(stderr,
                 "[mxfp] direct B200 step2 failed (%d), fallback to legacy step2 kernel\n",
                 static_cast<int>(st2));
    cudaGetLastError();
  }

  constexpr int kThreads = 128;
  int64_t n_out = static_cast<int64_t>(n_rows) * hidden_;
  int b3 = static_cast<int>((n_out + kThreads - 1) / kThreads);
  gemm2_scatter_accumulate_kernel<<<b3, kThreads, 0, stream>>>(
      c_perm_e, hidden_, intermediate_, block_, intermediate_blocks_, n_rows, permuted_tok_e,
      permuted_w_e, w2_e, s2_e, emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_,
      quantize_scale_e8m0_, out_acc_dev);
}

cudaError_t DeviceMxfpGemmModule::RunStep2AllExpertsDirect(const float* c_perm_all_dev,
                                                           const int* expert_t_valid,
                                                           const int* expert_offset,
                                                           const int* valid_token_idx,
                                                           const float* valid_token_w,
                                                           const uint8_t* gemm2_w_dev,
                                                           const float* gemm2_s_dev,
                                                           float* out_acc_dev,
                                                           cudaStream_t stream) const {
  if (!b200_direct_enabled_) return cudaErrorNotSupported;
  return b200::direct::LaunchStep2DirectAllExperts(c_perm_all_dev, expert_t_valid, expert_offset,
                                                   valid_token_idx, valid_token_w, gemm2_w_dev,
                                                   gemm2_s_dev, out_acc_dev, stream);
}

bool DeviceMxfpGemmModule::IsB200DirectEnabled() const { return b200_direct_enabled_; }

}  // namespace mxfp
