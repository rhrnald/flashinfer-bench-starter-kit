/*
 * TVM-FFI implementation for:
 *   gdn_prefill_qk4_v8_d128_k_last
 *
 * Correctness-first naive recurrence kernel for k-last state layout.
 *
 * Pseudocode:
 *
 * // Shapes
 * // Q         : [total_seq_len, QH, H]   , dtype = bfloat16
 * // K         : [total_seq_len, KH, H]   , dtype = bfloat16
 * // V         : [total_seq_len, VH, H]   , dtype = bfloat16
 * // state     : [num_seqs, VH, H, H]     , dtype = float32
 * // A_log     : [VH]                     , dtype = float32
 * // a         : [total_seq_len, VH]      , dtype = bfloat16
 * // dt_bias   : [VH]                     , dtype = float32
 * // b         : [total_seq_len, VH]      , dtype = bfloat16
 * // cu_seqlens: [num_seqs + 1]           , dtype = int32/int64
 * // output    : [total_seq_len, VH, H]   , dtype = bfloat16
 * // new_state : [num_seqs, VH, H, H]     , dtype = float32
 *
 * for (int seq = 0; seq < num_seqs; ++seq) {
 *   int start = cu_seqlens[seq];
 *   int end = cu_seqlens[seq + 1];
 *
 *   for (int vh = 0; vh < VH; ++vh) {
 *     int kh = vh / (VH / KH);
 *     int qh = vh / (VH / QH);
 *     S[vh, :, :] = state[seq, vh, :, :];
 *
 *     for (int tok = start; tok < end; ++tok) {
 *       float x = float(a[tok, vh]) + dt_bias[vh];
 *       float g = exp(-exp(A_log[vh]) * softplus(x));
 *       float beta = sigmoid(float(b[tok, vh]));
 *
 *       for (int i = 0; i < H; ++i) {
 *         S[vh, i, :] = g * S[vh, i, :];
 *         float old_v = dot(S[vh, i, :], K[tok, kh, :]);
 *         S[vh, i, :] =
 *             S[vh, i, :] +
 *             K[tok, kh, :] * (beta * (V[tok, vh, i] - old_v));
 *         output[tok, vh, i] = scale * dot(S[vh, i, :], Q[tok, qh, :]);
 *       }
 *     }
 *
 *     new_state[seq, vh, :, :] = S[vh, :, :];
 *   }
 * }
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
  constexpr int H = 128;

  int seq_head = blockIdx.x;
  int seq_idx = seq_head / VH;
  int vh = seq_head % VH;
  if (seq_idx >= num_seqs) return;

  int qh = vh / (VH / QH);
  int kh = vh / (VH / KH);

  int64_t seq_start = cu_seqlens[seq_idx];
  int64_t seq_end = cu_seqlens[seq_idx + 1];
  if (seq_end <= seq_start) return;

  __shared__ float q_sh[H];
  __shared__ float k_sh[H];

  // Initialize new_state from provided state (or zeros).
  for (int h_idx = threadIdx.x; h_idx < H; h_idx += blockDim.x) {
    for (int ki = 0; ki < H; ++ki) {
      int64_t idx = ((static_cast<int64_t>(seq_idx) * VH + vh) * H + h_idx) * H + ki;
      new_state[idx] = has_state ? state[idx] : 0.f;
    }
  }
  __syncthreads();

  for (int64_t t = seq_start; t < seq_end; ++t) {
    if (threadIdx.x < H) {
      q_sh[threadIdx.x] = __bfloat162float(q[(t * QH + qh) * H + threadIdx.x]);
      k_sh[threadIdx.x] = __bfloat162float(k[(t * KH + kh) * H + threadIdx.x]);
    }
    __syncthreads();

    float x = __bfloat162float(a[t * VH + vh]) + dt_bias[vh];
    float g = expf(-expf(A_log[vh]) * softplusf(x));
    float beta = sigmoidf(__bfloat162float(b[t * VH + vh]));

    for (int h_idx = threadIdx.x; h_idx < H; h_idx += blockDim.x) {
      // old_v = k @ (g * state)
      double old_v = 0.0;
      for (int ki = 0; ki < H; ++ki) {
        int64_t s_idx = ((static_cast<int64_t>(seq_idx) * VH + vh) * H + h_idx) * H + ki;
        old_v += static_cast<double>(k_sh[ki]) * static_cast<double>(g * new_state[s_idx]);
      }

      float vv = __bfloat162float(v[(t * VH + vh) * H + h_idx]);
      float delta = beta * (vv - static_cast<float>(old_v));

      // state <- g * state + k^T * delta
      for (int ki = 0; ki < H; ++ki) {
        int64_t s_idx = ((static_cast<int64_t>(seq_idx) * VH + vh) * H + h_idx) * H + ki;
        new_state[s_idx] = g * new_state[s_idx] + k_sh[ki] * delta;
      }

      // output = scale * (q @ state)
      double acc = 0.0;
      for (int ki = 0; ki < H; ++ki) {
        int64_t s_idx = ((static_cast<int64_t>(seq_idx) * VH + vh) * H + h_idx) * H + ki;
        acc += static_cast<double>(q_sh[ki]) * static_cast<double>(new_state[s_idx]);
      }
      output[(t * VH + vh) * H + h_idx] = __float2bfloat16(scale * static_cast<float>(acc));
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
