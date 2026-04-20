#pragma once

#include <cuda_runtime.h>
#include <cuda/ptx>

#include <cstdint>

namespace mxfp::b200::ptx {

using TmemHandle = uint32_t;

// NOTE(B200):
// - `ncols` here follows tcgen05.alloc/dealloc semantics.
// - Current callsites invoke this only from lane 0, so no block-wide sync is
//   used inside this helper.
__device__ __forceinline__ TmemHandle tmem_alloc(uint32_t ncols) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (ncols == 0) return 0;
  __shared__ uint32_t tmem_addr_smem;
  cuda::ptx::tcgen05_alloc(cuda::ptx::cta_group_1, &tmem_addr_smem, ncols);
  return tmem_addr_smem;
#else
  (void)ncols;
  return 0;
#endif
}

__device__ __forceinline__ void tmem_dealloc(TmemHandle h, uint32_t ncols = 1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (h != 0 && ncols != 0) {
    cuda::ptx::tcgen05_dealloc(cuda::ptx::cta_group_1, h, ncols);
  }
#else
  (void)h;
  (void)ncols;
#endif
}

__device__ __forceinline__ void tmem_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  cuda::ptx::tcgen05_wait_ld();
  cuda::ptx::tcgen05_wait_st();
#endif
}

__device__ __forceinline__ void tmem_cp() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  cuda::ptx::tcgen05_fence_before_thread_sync();
  cuda::ptx::tcgen05_fence_after_thread_sync();
#endif
}

}  // namespace mxfp::b200::ptx
