/*
 * Placeholder implementation for:
 *   moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
 *
 * Goal: provide a compilable/runnable TVM-FFI entry point.
 * TODO: replace with real kernel.
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

}  // namespace

void moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048_impl(
    TensorView routing_logits, TensorView routing_bias, TensorView hidden_states,
    TensorView hidden_states_scale, TensorView gemm1_weights, TensorView gemm1_weights_scale,
    TensorView gemm2_weights, TensorView gemm2_weights_scale, int64_t local_expert_offset,
    double routed_scaling_factor, TensorView output) {
  (void)routing_logits;
  (void)routing_bias;
  (void)hidden_states;
  (void)hidden_states_scale;
  (void)gemm1_weights;
  (void)gemm1_weights_scale;
  (void)gemm2_weights;
  (void)gemm2_weights_scale;
  (void)local_expert_offset;
  (void)routed_scaling_factor;

  TVM_FFI_ICHECK_EQ(output.ndim(), 2);
  int64_t n = output.size(0) * output.size(1);

  DLDevice dev = output.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  fill_bf16_zero<<<blocks, kThreads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(output.data_ptr()), n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048,
                              moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048_impl);
