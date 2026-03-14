/*
 * TVM-FFI entry for:
 *   gdn_decode_qk4_v8_d128_k_last
 *
 * Optimized decode kernel for the public TP=4 contest spec.
 *
 * Pseudocode:
 *
 * // Shapes
 * // Q        : [Batch, 1, QH, H]
 * // K        : [Batch, 1, KH, H]
 * // V        : [Batch, 1, VH, H]
 * // state    : [Batch, VH, H, H]
 * // a        : [Batch, 1, VH]
 * // b        : [Batch, 1, VH]
 * // A_log    : [VH]
 * // dt_bias  : [VH]
 * // output   : [Batch, 1, VH, H]
 * // new_state: [Batch, VH, H, H]
 *
 * for (int b = 0; b < Batch; ++b) {
 *   for (int vh = 0; vh < VH; ++vh) {
 *     int kh = vh / (VH / KH);
 *     int qh = vh / (VH / QH);
 *
 *     int a_idx = (b * 1 + 0) * VH + vh;
 *     float x = float(a[b, 0, vh]) + dt_bias[vh];
 *     float g = exp(-exp(A_log[vh]) * softplus(x));
 *     float beta = sigmoid(float(bias_b[b, 0, vh]));
 *
 *     for (int i = 0; i < H; ++i) {
 *       new_state[b, vh, i, :] = g * state[b, vh, i, :];
 *       float old_v = dot(new_state[b, vh, i, :], K[b, 0, kh, :]);
 *       new_state[b, vh, i, :] =
 *           new_state[b, vh, i, :] +
 *           K[b, 0, kh, :] * (beta * (V[b, 0, vh, i] - old_v));
 *       output[b, 0, vh, i] = scale * dot(new_state[b, vh, i, :], Q[b, 0, qh, :]);
 *     }
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

__device__ __forceinline__ float warp_sum(float x) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
  return x;
}

__global__ void gdn_decode_qk4_v8_d128_k_last_kernel_v3_831(
    const __nv_bfloat16* q,      // [B,T,QH,H] = [B,1,4,128]
    const __nv_bfloat16* k,      // [B,T,KH,H] = [B,1,4,128]
    const __nv_bfloat16* v,      // [B,T,VH,H] = [B,1,8,128]
    const float* state,          // [B,VH,H,H] = [B,8,128,128]
    const float* A_log,          // [VH] = [8]
    const __nv_bfloat16* a,      // [B,T,VH] = [B,1,8]
    const float* dt_bias,        // [VH] = [8]
    const __nv_bfloat16* b,      // [B,T,VH] = [B,1,8]
    float scale,                 // [] = scalar
    __nv_bfloat16* output,       // [B,T,VH,H] = [B,1,8,128]
    float* new_state,            // [B,VH,H,H] = [B,8,128,128]
    int batch_size) {            // [1] = B
  constexpr int QH = 4;
  constexpr int KH = 4;
  constexpr int VH = 8;
  constexpr int H = 128;
  constexpr int T = 1;
  constexpr int WARPS = H / 32;

  __shared__ float sh_q[H];
  __shared__ float sh_k[H];
  __shared__ float warp_sums[WARPS];
  __shared__ float sh_g;
  __shared__ float sh_beta;
  __shared__ float sh_old_v;

  int block = blockIdx.x;
  int bid = block / (VH * H);
  int rem = block % (VH * H);
  int vh = rem / H;
  int row = rem % H;
  if (bid >= batch_size) return;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  int qh = vh / (VH / QH);
  int kh = vh / (VH / KH);

  int q_idx = ((bid * T + 0) * QH + qh) * H + tid;
  int k_idx = ((bid * T + 0) * KH + kh) * H + tid;
  sh_q[tid] = __bfloat162float(q[q_idx]);
  sh_k[tid] = __bfloat162float(k[k_idx]);

  if (tid == 0) {
    int a_idx = (bid * T + 0) * VH + vh;
    float x = __bfloat162float(a[a_idx]) + dt_bias[vh];
    sh_g = expf(-expf(A_log[vh]) * softplusf(x));
    sh_beta = sigmoidf(__bfloat162float(b[a_idx]));
  }
  __syncthreads();

  int row_base = ((bid * VH + vh) * H + row) * H;
  float state_val = state[row_base + tid];
  float decayed = sh_g * state_val;

  float old_v_partial = decayed * sh_k[tid];
  float old_v_warp = warp_sum(old_v_partial);
  if (lane == 0) {
    warp_sums[warp_id] = old_v_warp;
  }
  __syncthreads();

  if (warp_id == 0) {
    float x = (lane < WARPS) ? warp_sums[lane] : 0.f;
    x = warp_sum(x);
    if (lane == 0) {
      sh_old_v = x;
    }
  }
  __syncthreads();

  int v_idx = ((bid * T + 0) * VH + vh) * H + row;
  float vv = __bfloat162float(v[v_idx]);
  float delta = sh_beta * (vv - sh_old_v);

  float new_state_val = decayed + sh_k[tid] * delta;
  new_state[row_base + tid] = new_state_val;

  float acc_partial = sh_q[tid] * new_state_val;
  float acc_warp = warp_sum(acc_partial);
  if (lane == 0) {
    warp_sums[warp_id] = acc_warp;
  }
  __syncthreads();

  if (warp_id == 0) {
    float x = (lane < WARPS) ? warp_sums[lane] : 0.f;
    x = warp_sum(x);
    if (lane == 0) {
      int out_idx = ((bid * T + 0) * VH + vh) * H + row;
      output[out_idx] = __float2bfloat16(scale * x);
    }
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
  gdn_decode_qk4_v8_d128_k_last_kernel_v3_831<<<batch_size * 8 * 128, kThreads, 0, stream>>>(
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
