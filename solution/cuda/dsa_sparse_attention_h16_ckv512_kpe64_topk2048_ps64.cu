/*
 * Placeholder implementation for:
 *   dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
 *
 * Goal: provide a compilable/runnable TVM-FFI entry point.
 * TODO: replace with real kernel.
 *
 * Pseudocode:
 *
 * // Axes / constants
 * // num_tokens     : variable
 * // num_qo_heads   : 16
 * // head_dim_ckv   : 512
 * // head_dim_kpe   : 64
 * // page_size      : 64
 * // topk           : 2048
 * // num_pages      : variable
 *
 * // Shapes
 * // q_nope         : [num_tokens, num_qo_heads, head_dim_ckv]   , dtype = bfloat16
 * // q_pe           : [num_tokens, num_qo_heads, head_dim_kpe]   , dtype = bfloat16
 * // ckv_cache      : [num_pages, page_size, head_dim_ckv]       , dtype = bfloat16
 * // kpe_cache      : [num_pages, page_size, head_dim_kpe]       , dtype = bfloat16
 * // sparse_indices : [num_tokens, topk]                         , dtype = int32
 * // sm_scale       : scalar                                     , dtype = float32
 * // output         : [num_tokens, num_qo_heads, head_dim_ckv]   , dtype = bfloat16
 * // lse            : [num_tokens, num_qo_heads]                 , dtype = float32
 *
 * Kc_all = reshape(ckv_cache, [num_pages * page_size, head_dim_ckv]);
 * Kp_all = reshape(kpe_cache, [num_pages * page_size, head_dim_kpe]);
 *
 * for (int t = 0; t < num_tokens; ++t) {
 *   int indices[topk] = sparse_indices[t];
 *   valid_indices = filter(indices != -1);
 *
 *   if (valid_indices.size() == 0) {
 *     output[t, :, :] = 0;
 *     lse[t, :] = -inf;
 *     continue;
 *   }
 *
 *   Kc = Kc_all[valid_indices, :];
 *   Kp = Kp_all[valid_indices, :];
 *   qn = q_nope[t, :, :];
 *   qp = q_pe[t, :, :];
 *
 *   for (int h = 0; h < num_qo_heads; ++h) {
 *     for (int j = 0; j < num_valid; ++j) {
 *       logits[h, j] = dot(qn[h, :], Kc[j, :]) + dot(qp[h, :], Kp[j, :]);
 *       logits_scaled[h, j] = sm_scale * logits[h, j];
 *     }
 *
 *     lse[t, h] = logsumexp(logits_scaled[h, :]) / log(2.0);
 *     attn[h, :] = softmax(logits_scaled[h, :]);
 *
 *     output_fp32[h, :] = 0;
 *     for (int j = 0; j < num_valid; ++j) {
 *       output_fp32[h, :] += attn[h, j] * Kc[j, :];
 *     }
 *
 *     output[t, h, :] = cast_bfloat16(output_fp32[h, :]);
 *   }
 * }
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

using tvm::ffi::TensorView;

namespace {

__global__ void fill_bf16_zero(__nv_bfloat16* out, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = __float2bfloat16(0.0f);
}

__global__ void fill_f32_zero(float* out, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = 0.0f;
}

}  // namespace

void dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_impl(
    TensorView q_nope, TensorView q_pe, TensorView ckv_cache, TensorView kpe_cache,
    TensorView sparse_indices, double sm_scale, TensorView output, TensorView lse) {
  (void)q_nope;
  (void)q_pe;
  (void)ckv_cache;
  (void)kpe_cache;
  (void)sparse_indices;
  (void)sm_scale;

  TVM_FFI_ICHECK_EQ(output.ndim(), 3);
  TVM_FFI_ICHECK_EQ(lse.ndim(), 2);

  int64_t out_n = output.size(0) * output.size(1) * output.size(2);
  int64_t lse_n = lse.size(0) * lse.size(1);

  DLDevice dev = output.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  constexpr int kThreads = 256;
  int out_blocks = static_cast<int>((out_n + kThreads - 1) / kThreads);
  int lse_blocks = static_cast<int>((lse_n + kThreads - 1) / kThreads);

  fill_bf16_zero<<<out_blocks, kThreads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(output.data_ptr()), out_n);
  fill_f32_zero<<<lse_blocks, kThreads, 0, stream>>>(static_cast<float*>(lse.data_ptr()), lse_n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64,
                              dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_impl);
