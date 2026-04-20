#pragma once

#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cuda.h>

#include <cstdint>

namespace mxfp::b200::ptx {

struct TmaDesc {
  const void* tensor_map = nullptr;  // device pointer to encoded CUtensorMap
  uint64_t* smem_bar = nullptr;      // CTA-shared mbarrier pointer
  int cta_group = 1;                 // 1 or 2
  bool valid = false;
};

__device__ __forceinline__ void tma_async_load_2d(const TmaDesc& desc,
                                                  void* smem_dst,
                                                  int i0,
                                                  int i1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (!desc.valid || desc.tensor_map == nullptr || desc.smem_bar == nullptr || smem_dst == nullptr) return;
  const int32_t coords[2] = {static_cast<int32_t>(i0), static_cast<int32_t>(i1)};
  if (desc.cta_group == 2) {
    cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
                                    cuda::ptx::cta_group_2, smem_dst, desc.tensor_map, coords,
                                    desc.smem_bar);
  } else {
    cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
                                    cuda::ptx::cta_group_1, smem_dst, desc.tensor_map, coords,
                                    desc.smem_bar);
  }
#else
  (void)desc;
  (void)smem_dst;
  (void)i0;
  (void)i1;
#endif
}

__device__ __forceinline__ void tma_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  cuda::ptx::cp_async_bulk_commit_group();
#endif
}

__device__ __forceinline__ void tma_wait_group(int n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  switch (n) {
    case 0: cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{}); break;
    case 1: cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<1>{}); break;
    case 2: cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<2>{}); break;
    case 3: cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<3>{}); break;
    case 4: cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<4>{}); break;
    case 5: cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<5>{}); break;
    case 6: cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<6>{}); break;
    case 7: cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<7>{}); break;
    default: cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{}); break;
  }
#else
  (void)n;
#endif
}

inline CUresult EncodeTensorMap2D(CUtensorMap* out_tensor_map,
                                  CUtensorMapDataType dtype,
                                  void* global_address,
                                  uint64_t rows,
                                  uint64_t cols,
                                  uint64_t row_stride_bytes,
                                  uint32_t box_rows,
                                  uint32_t box_cols) {
  if (out_tensor_map == nullptr || global_address == nullptr) return CUDA_ERROR_INVALID_VALUE;
  const cuuint64_t global_dim[2] = {rows, cols};
  const cuuint64_t global_strides[1] = {row_stride_bytes};
  const cuuint32_t box_dim[2] = {box_rows, box_cols};
  const cuuint32_t element_strides[2] = {1u, 1u};
  return cuTensorMapEncodeTiled(out_tensor_map, dtype, 2, global_address, global_dim, global_strides,
                                box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

}  // namespace mxfp::b200::ptx
