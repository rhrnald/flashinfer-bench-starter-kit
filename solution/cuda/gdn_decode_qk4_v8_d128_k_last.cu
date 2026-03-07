/*
 * TVM-FFI entry for:
 *   gdn_decode_qk4_v8_d128_k_last
 *
 * This file defines an exported host entry point so build/runtime wiring works.
 * Naive correctness-first implementation for decode (slow but straightforward).
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

using tvm::ffi::TensorView;

__device__ __forceinline__ float softplusf(float x) { return log1pf(expf(x)); }

__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

__global__ void gdn_decode_naive_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                        const __nv_bfloat16* v, const float* state,
                                        const float* A_log, const __nv_bfloat16* a,
                                        const float* dt_bias, const __nv_bfloat16* b, float scale,
                                        __nv_bfloat16* output, float* new_state, int batch_size) {
  constexpr int QH = 4;
  constexpr int KH = 4;
  constexpr int VH = 8;
  constexpr int D = 128;
  constexpr int T = 1;

  int bh = blockIdx.x;
  int bid = bh / VH;
  int vh = bh % VH;
  if (bid >= batch_size) return;

  int qh = vh / (VH / QH);
  int kh = vh / (VH / KH);

  int a_idx = (bid * T + 0) * VH + vh;
  float x = __bfloat162float(a[a_idx]) + dt_bias[vh];
  float g = expf(-expf(A_log[vh]) * softplusf(x));
  float beta = sigmoidf(__bfloat162float(b[a_idx]));

  // Parallelize over vi (V dimension) within block.
  for (int vi = threadIdx.x; vi < D; vi += blockDim.x) {
    float old_v = 0.f;
    for (int ki = 0; ki < D; ++ki) {
      int k_idx = ((bid * T + 0) * KH + kh) * D + ki;
      int s_idx = ((bid * VH + vh) * D + vi) * D + ki;  // [B,H,V,K]
      float kf = __bfloat162float(k[k_idx]);
      old_v += kf * (g * state[s_idx]);
    }

    int v_idx = ((bid * T + 0) * VH + vh) * D + vi;
    float vv = __bfloat162float(v[v_idx]);
    float delta = beta * (vv - old_v);

    for (int ki = 0; ki < D; ++ki) {
      int k_idx = ((bid * T + 0) * KH + kh) * D + ki;
      int s_idx = ((bid * VH + vh) * D + vi) * D + ki;
      float kf = __bfloat162float(k[k_idx]);
      new_state[s_idx] = g * state[s_idx] + kf * delta;
    }
  }

  // output[v_i] = scale * sum_k q[k] * new_state[v_i, k]
  for (int vi = threadIdx.x; vi < D; vi += blockDim.x) {
    float acc = 0.f;
    for (int ki = 0; ki < D; ++ki) {
      int q_idx = ((bid * T + 0) * QH + qh) * D + ki;
      int ns_idx = ((bid * VH + vh) * D + vi) * D + ki;
      float qf = __bfloat162float(q[q_idx]);
      acc += qf * new_state[ns_idx];
    }
    int out_idx = ((bid * T + 0) * VH + vh) * D + vi;
    output[out_idx] = __float2bfloat16(scale * acc);
  }
}

void gdn_decode_qk4_v8_d128_k_last_impl(TensorView q, TensorView k, TensorView v, TensorView state,
                                        TensorView A_log, TensorView a, TensorView dt_bias,
                                        TensorView b, double scale, TensorView output,
                                        TensorView new_state) {
  TVM_FFI_ICHECK_EQ(q.ndim(), 4);
  TVM_FFI_ICHECK_EQ(k.ndim(), 4);
  TVM_FFI_ICHECK_EQ(v.ndim(), 4);
  TVM_FFI_ICHECK_EQ(state.ndim(), 4);
  TVM_FFI_ICHECK_EQ(output.ndim(), 4);
  TVM_FFI_ICHECK_EQ(new_state.ndim(), 4);
  TVM_FFI_ICHECK_EQ(q.size(2), 4);
  TVM_FFI_ICHECK_EQ(k.size(2), 4);
  TVM_FFI_ICHECK_EQ(v.size(2), 8);
  TVM_FFI_ICHECK_EQ(q.size(3), 128);
  TVM_FFI_ICHECK_EQ(k.size(3), 128);
  TVM_FFI_ICHECK_EQ(v.size(3), 128);

  int batch_size = static_cast<int>(q.size(0));
  DLDevice dev = q.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  constexpr int kThreads = 128;
  gdn_decode_naive_kernel<<<batch_size * 8, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q.data_ptr()),
      static_cast<const __nv_bfloat16*>(k.data_ptr()),
      static_cast<const __nv_bfloat16*>(v.data_ptr()), static_cast<const float*>(state.data_ptr()),
      static_cast<const float*>(A_log.data_ptr()), static_cast<const __nv_bfloat16*>(a.data_ptr()),
      static_cast<const float*>(dt_bias.data_ptr()),
      static_cast<const __nv_bfloat16*>(b.data_ptr()), static_cast<float>(scale),
      static_cast<__nv_bfloat16*>(output.data_ptr()), static_cast<float*>(new_state.data_ptr()),
      batch_size);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_decode_qk4_v8_d128_k_last,
                              gdn_decode_qk4_v8_d128_k_last_impl);
