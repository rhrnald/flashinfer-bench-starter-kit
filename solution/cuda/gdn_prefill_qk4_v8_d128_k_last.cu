/*
 * TVM-FFI implementation for:
 *   gdn_prefill_qk4_v8_d128_k_last
 *
 * Correctness-first naive recurrence kernel for k-last state layout.
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

__global__ void gdn_prefill_naive_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const float* state,
    const float* A_log, const __nv_bfloat16* a, const float* dt_bias, const __nv_bfloat16* b,
    const int64_t* cu_seqlens, float scale, __nv_bfloat16* output, float* new_state, int num_seqs,
    bool has_state) {
  constexpr int QH = 4;
  constexpr int KH = 4;
  constexpr int VH = 8;
  constexpr int D = 128;

  int seq_head = blockIdx.x;
  int seq_idx = seq_head / VH;
  int vh = seq_head % VH;
  if (seq_idx >= num_seqs) return;

  int qh = vh / (VH / QH);
  int kh = vh / (VH / KH);

  int64_t seq_start = cu_seqlens[seq_idx];
  int64_t seq_end = cu_seqlens[seq_idx + 1];
  if (seq_end <= seq_start) return;

  __shared__ float q_sh[D];
  __shared__ float k_sh[D];

  // Initialize new_state from provided state (or zeros).
  for (int vi = threadIdx.x; vi < D; vi += blockDim.x) {
    for (int ki = 0; ki < D; ++ki) {
      int64_t idx = ((static_cast<int64_t>(seq_idx) * VH + vh) * D + vi) * D + ki;
      new_state[idx] = has_state ? state[idx] : 0.f;
    }
  }
  __syncthreads();

  for (int64_t t = seq_start; t < seq_end; ++t) {
    if (threadIdx.x < D) {
      q_sh[threadIdx.x] = __bfloat162float(q[(t * QH + qh) * D + threadIdx.x]);
      k_sh[threadIdx.x] = __bfloat162float(k[(t * KH + kh) * D + threadIdx.x]);
    }
    __syncthreads();

    float x = __bfloat162float(a[t * VH + vh]) + dt_bias[vh];
    float g = expf(-expf(A_log[vh]) * softplusf(x));
    float beta = sigmoidf(__bfloat162float(b[t * VH + vh]));

    for (int vi = threadIdx.x; vi < D; vi += blockDim.x) {
      // old_v = k @ (g * state)
      double old_v = 0.0;
      for (int ki = 0; ki < D; ++ki) {
        int64_t s_idx = ((static_cast<int64_t>(seq_idx) * VH + vh) * D + vi) * D + ki;
        old_v += static_cast<double>(k_sh[ki]) * static_cast<double>(g * new_state[s_idx]);
      }

      float vv = __bfloat162float(v[(t * VH + vh) * D + vi]);
      float delta = beta * (vv - static_cast<float>(old_v));

      // state <- g * state + k^T * delta
      for (int ki = 0; ki < D; ++ki) {
        int64_t s_idx = ((static_cast<int64_t>(seq_idx) * VH + vh) * D + vi) * D + ki;
        new_state[s_idx] = g * new_state[s_idx] + k_sh[ki] * delta;
      }

      // output = scale * (q @ state)
      double acc = 0.0;
      for (int ki = 0; ki < D; ++ki) {
        int64_t s_idx = ((static_cast<int64_t>(seq_idx) * VH + vh) * D + vi) * D + ki;
        acc += static_cast<double>(q_sh[ki]) * static_cast<double>(new_state[s_idx]);
      }
      output[(t * VH + vh) * D + vi] = __float2bfloat16(scale * static_cast<float>(acc));
    }
    __syncthreads();
  }
}

void gdn_prefill_qk4_v8_d128_k_last_impl(TensorView q, TensorView k, TensorView v, TensorView state,
                                         TensorView A_log, TensorView a, TensorView dt_bias,
                                         TensorView b, TensorView cu_seqlens, double scale,
                                         TensorView output, TensorView new_state) {
  TVM_FFI_ICHECK_EQ(q.ndim(), 3);
  TVM_FFI_ICHECK_EQ(k.ndim(), 3);
  TVM_FFI_ICHECK_EQ(v.ndim(), 3);
  TVM_FFI_ICHECK_EQ(A_log.ndim(), 1);
  TVM_FFI_ICHECK_EQ(a.ndim(), 2);
  TVM_FFI_ICHECK_EQ(dt_bias.ndim(), 1);
  TVM_FFI_ICHECK_EQ(b.ndim(), 2);
  TVM_FFI_ICHECK_EQ(cu_seqlens.ndim(), 1);
  TVM_FFI_ICHECK_EQ(output.ndim(), 3);
  TVM_FFI_ICHECK_EQ(new_state.ndim(), 4);

  TVM_FFI_ICHECK_EQ(q.size(1), 4);
  TVM_FFI_ICHECK_EQ(k.size(1), 4);
  TVM_FFI_ICHECK_EQ(v.size(1), 8);
  TVM_FFI_ICHECK_EQ(q.size(2), 128);
  TVM_FFI_ICHECK_EQ(k.size(2), 128);
  TVM_FFI_ICHECK_EQ(v.size(2), 128);

  int num_seqs = static_cast<int>(new_state.size(0));
  bool has_state = state.ndim() == 4 && state.data_ptr() != nullptr;

  DLDevice dev = q.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  constexpr int kThreads = 128;
  int blocks = num_seqs * 8;  // [num_seqs, num_v_heads]
  gdn_prefill_naive_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q.data_ptr()),
      static_cast<const __nv_bfloat16*>(k.data_ptr()),
      static_cast<const __nv_bfloat16*>(v.data_ptr()),
      has_state ? static_cast<const float*>(state.data_ptr()) : nullptr,
      static_cast<const float*>(A_log.data_ptr()),
      static_cast<const __nv_bfloat16*>(a.data_ptr()),
      static_cast<const float*>(dt_bias.data_ptr()),
      static_cast<const __nv_bfloat16*>(b.data_ptr()),
      static_cast<const int64_t*>(cu_seqlens.data_ptr()), static_cast<float>(scale),
      static_cast<__nv_bfloat16*>(output.data_ptr()), static_cast<float*>(new_state.data_ptr()),
      num_seqs, has_state);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_prefill_qk4_v8_d128_k_last,
                              gdn_prefill_qk4_v8_d128_k_last_impl);
