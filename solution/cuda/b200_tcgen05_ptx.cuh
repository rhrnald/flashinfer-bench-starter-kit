#pragma once

#include <cuda_runtime.h>
#include <cuda/ptx>

#include <cstdint>

namespace mxfp::b200::ptx {

struct Tcgen05MmaDesc {
  uint32_t d_tmem = 0;
  uint64_t a_desc = 0;
  uint64_t b_desc = 0;
  uint32_t i_desc = 0;
  int cta_group = 1;  // 1 or 2
  bool enable_input_d = false;
  bool valid = false;
  uint32_t disable_output_lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

__device__ __forceinline__ void tcgen05_mma_f8(const Tcgen05MmaDesc& d) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (!d.valid) return;
  if (d.cta_group == 2) {
    cuda::ptx::tcgen05_mma(cuda::ptx::kind_f8f6f4, cuda::ptx::cta_group_2_t{}, d.d_tmem, d.a_desc,
                           d.b_desc, d.i_desc, d.disable_output_lane, d.enable_input_d);
  } else {
    const uint32_t disable_output_lane_4[4] = {
        d.disable_output_lane[0],
        d.disable_output_lane[1],
        d.disable_output_lane[2],
        d.disable_output_lane[3],
    };
    cuda::ptx::tcgen05_mma(cuda::ptx::kind_f8f6f4, cuda::ptx::cta_group_1_t{}, d.d_tmem, d.a_desc,
                           d.b_desc, d.i_desc, disable_output_lane_4, d.enable_input_d);
  }
#else
  (void)d;
#endif
}

}  // namespace mxfp::b200::ptx
