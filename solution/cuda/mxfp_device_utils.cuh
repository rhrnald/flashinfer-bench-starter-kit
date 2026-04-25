#pragma once

#include <cuda_fp16.h>
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define FIB_HAS_CUDA_FP8 1
#else
#define FIB_HAS_CUDA_FP8 0
#endif

#include <cstdint>
#include <cstdlib>
#include <math.h>

namespace mxfp::detail {

inline int env_int_or_default(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  return std::atoi(value);
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

__device__ __forceinline__ float siluf_device(float x) { return x / (1.0f + __expf(-x)); }

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
#if FIB_HAS_CUDA_FP8
  __nv_fp8_e4m3 y(x);
  return *reinterpret_cast<uint8_t*>(&y);
#else
  (void)x;
  return 0;
#endif
}

__device__ __forceinline__ float fp8_native_to_float_device(uint8_t x) {
#if FIB_HAS_CUDA_FP8
  __nv_fp8_e4m3 y = *reinterpret_cast<__nv_fp8_e4m3*>(&x);
  return static_cast<float>(y);
#else
  (void)x;
  return 0.0f;
#endif
}

// Optional FP8-unit emulation: quantize intermediate values to an E4M3FN-like grid.
__device__ __forceinline__ float quantize_e4m3fn_like(float x) {
  if (!isfinite(x) || x == 0.0f) return 0.0f;
  float sign = x < 0.0f ? -1.0f : 1.0f;
  float ax = fabsf(x);
  const float kMax = 448.0f;
  const float kMinSub = ldexpf(1.0f, -9);
  if (ax >= kMax) return sign * kMax;
  if (ax < kMinSub) return 0.0f;

  int e2;
  float m = frexpf(ax, &e2);
  (void)m;
  int e = e2 - 1;

  if (e < -6) {
    float q = nearbyintf(ax / kMinSub);
    q = fminf(7.0f, fmaxf(0.0f, q));
    return sign * q * kMinSub;
  }
  if (e > 8) return sign * kMax;

  float base = ldexpf(1.0f, e);
  float mf = ax / base;
  float qm = nearbyintf((mf - 1.0f) * 8.0f);
  qm = fminf(7.0f, fmaxf(0.0f, qm));
  float qmf = 1.0f + qm * 0.125f;
  return sign * qmf * base;
}

}  // namespace mxfp::detail
