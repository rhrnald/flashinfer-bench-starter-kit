#include "mxfp_gemm_module.h"

#include <cuda_fp16.h>

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
      g1_dev_(nullptr),
      c_dev_(nullptr) {
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
  if (emulate_fp8_unit_ || emulate_fp16_operands_ || emulate_acc_half_) {
    std::fflush(stderr);
  }
}

DeviceMxfpGemmModule::~DeviceMxfpGemmModule() {
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

}  // namespace mxfp
