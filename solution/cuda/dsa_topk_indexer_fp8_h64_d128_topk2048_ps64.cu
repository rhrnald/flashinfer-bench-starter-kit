/*
 * Placeholder implementation for:
 *   dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
 *
 * Goal: provide a compilable/runnable TVM-FFI entry point.
 * TODO: replace with real kernel.
 */

#include <cuda_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

using tvm::ffi::TensorView;

namespace {

__global__ void fill_i32_neg1(int* out, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = -1;
}

}  // namespace

void dsa_topk_indexer_fp8_h64_d128_topk2048_ps64_impl(
    TensorView q_index_fp8, TensorView k_index_cache_fp8, TensorView weights,
    TensorView seq_lens, TensorView block_table, TensorView topk_indices) {
  (void)q_index_fp8;
  (void)k_index_cache_fp8;
  (void)weights;
  (void)seq_lens;
  (void)block_table;

  TVM_FFI_ICHECK_EQ(topk_indices.ndim(), 2);

  int64_t n = topk_indices.size(0) * topk_indices.size(1);

  DLDevice dev = topk_indices.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((n + kThreads - 1) / kThreads);

  fill_i32_neg1<<<blocks, kThreads, 0, stream>>>(static_cast<int*>(topk_indices.data_ptr()), n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsa_topk_indexer_fp8_h64_d128_topk2048_ps64,
                              dsa_topk_indexer_fp8_h64_d128_topk2048_ps64_impl);
