#pragma once

#include "b200_tcgen05_ptx.cuh"
#include "b200_tma_ptx.cuh"
#include "b200_tmem_ptx.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace mxfp::b200::direct {

static constexpr int kStep2Hidden = 7168;
static constexpr int kStep2Intermediate = 2048;
static constexpr int kStep2Block = 128;
static constexpr int kStep2LocalExperts = 32;

__device__ __forceinline__ float fp8_e4m3fn_to_float_device_step2(uint8_t x) {
  int sign = (x & 0x80) ? -1 : 1;
  int exp = (x >> 3) & 0x0f;
  int mant = x & 0x07;

  if (exp == 0) {
    if (mant == 0) return sign == 1 ? 0.0f : -0.0f;
    float frac = static_cast<float>(mant) * 0.125f;
    return sign * ldexpf(frac, -6);
  }

  float frac = 1.0f + static_cast<float>(mant) * 0.125f;
  return sign * ldexpf(frac, exp - 7);
}

// Step2 direct path:
// - one CTA computes [row, h-tile]
// - thread computes one (row, h) output and scatters with token routing weight
// - writes use atomicAdd for safety when execution policy changes to concurrent experts
__global__ void step2_gemm2_scatter_direct_kernel(const float* __restrict__ c_perm_dev,
                                                  int n_rows,
                                                  const int* __restrict__ permuted_tok_e,
                                                  const float* __restrict__ permuted_w_e,
                                                  const uint8_t* __restrict__ w2_e,
                                                  const float* __restrict__ s2_e,
                                                  float* __restrict__ out_acc_dev) {
  const int row = blockIdx.x;
  const int hb = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= n_rows) return;
  const int h = hb * kStep2Block + lane;
  if (h >= kStep2Hidden) return;

  if (lane == 0) {
    ptx::TmaDesc tma_desc{};
    ptx::tma_async_load_2d(tma_desc, nullptr, 0, 0);
    auto tmem_h = ptx::tmem_alloc(0);
    ptx::Tcgen05MmaDesc mma_desc{};
    ptx::tcgen05_mma_f8(mma_desc);
    ptx::tmem_dealloc(tmem_h);
  }

  const int intermediate_blocks = kStep2Intermediate / kStep2Block;
  const int tok = permuted_tok_e[row];
  const float token_w = permuted_w_e[row];
  const float* c_row = c_perm_dev + static_cast<int64_t>(row) * kStep2Intermediate;
  const uint8_t* w_row = w2_e + static_cast<int64_t>(h) * kStep2Intermediate;

  float acc = 0.0f;
  for (int ib = 0; ib < intermediate_blocks; ++ib) {
    const float scale = s2_e[hb * intermediate_blocks + ib];
    const int i0 = ib * kStep2Block;
    float raw = 0.0f;
    for (int u = 0; u < kStep2Block; ++u) {
      const int i = i0 + u;
      raw += c_row[i] * fp8_e4m3fn_to_float_device_step2(w_row[i]);
    }
    acc += raw * scale;
  }

  atomicAdd(&out_acc_dev[static_cast<int64_t>(tok) * kStep2Hidden + h], token_w * acc);
}

static inline cudaError_t LaunchStep2Direct(const float* c_perm_dev,
                                            int n_rows,
                                            const int* permuted_tok_e,
                                            const float* permuted_w_e,
                                            const uint8_t* w2_e,
                                            const float* s2_e,
                                            float* out_acc_dev,
                                            cudaStream_t stream) {
  if (n_rows <= 0) return cudaSuccess;
  dim3 grid(n_rows, (kStep2Hidden + kStep2Block - 1) / kStep2Block);
  dim3 threads(kStep2Block);
  step2_gemm2_scatter_direct_kernel<<<grid, threads, 0, stream>>>(
      c_perm_dev, n_rows, permuted_tok_e, permuted_w_e, w2_e, s2_e, out_acc_dev);
  return cudaGetLastError();
}

// Step2 all-experts direct path:
// - grid.x = H/128 (56 tiles), grid.y = 32 local experts
// - blockDim.x = 32, each lane computes 4 output channels
// - each CTA iterates all routed rows for its expert
// - writes to out_acc use atomicAdd because experts run concurrently and may
//   update the same (tok, h) location
__global__ void step2_gemm2_scatter_all_experts_direct_kernel(
    const float* __restrict__ c_perm_all_dev, const int* __restrict__ expert_t_valid,
    const int* __restrict__ expert_offset, const int* __restrict__ valid_token_idx,
    const float* __restrict__ valid_token_w, const uint8_t* __restrict__ w2_all_dev,
    const float* __restrict__ s2_all_dev, float* __restrict__ out_acc_dev) {
  const int hb = blockIdx.x;
  const int expert = blockIdx.y;
  const int lane = threadIdx.x;
  if (lane >= 32) return;
  if (expert >= kStep2LocalExperts) return;
  if (hb >= (kStep2Hidden / kStep2Block)) return;

  const int t_valid = expert_t_valid[expert];
  if (t_valid <= 0) return;
  const int row_start = expert_offset[expert];

  const int intermediate_blocks = kStep2Intermediate / kStep2Block;
  const size_t w2_expert_elems = static_cast<size_t>(kStep2Hidden) * kStep2Intermediate;
  const size_t s2_expert_elems =
      static_cast<size_t>(kStep2Hidden / kStep2Block) * intermediate_blocks;
  const uint8_t* w2_e = w2_all_dev + static_cast<size_t>(expert) * w2_expert_elems;
  const float* s2_e = s2_all_dev + static_cast<size_t>(expert) * s2_expert_elems;

  for (int row_local = 0; row_local < t_valid; ++row_local) {
    const int slot = row_start + row_local;
    const int tok = valid_token_idx[slot];
    const float tok_w = valid_token_w[slot];
    const float* c_row = c_perm_all_dev + static_cast<int64_t>(slot) * kStep2Intermediate;

    float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int ib = 0; ib < intermediate_blocks; ++ib) {
      if (lane == 0) {
        // TODO(B200): Replace with real Step2 TMA/tcgen05 staging sequence.
        ptx::TmaDesc tma_desc{};
        ptx::tma_async_load_2d(tma_desc, nullptr, hb, ib);
        auto tmem_h = ptx::tmem_alloc(0);
        ptx::Tcgen05MmaDesc mma_desc{};
        ptx::tcgen05_mma_f8(mma_desc);
        ptx::tmem_dealloc(tmem_h);
      }

      const float scale = s2_e[hb * intermediate_blocks + ib];
      const int i0 = ib * kStep2Block;
      float raw4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int u = 0; u < kStep2Block; ++u) {
        const int i = i0 + u;
        const float cv = c_row[i];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const int h = hb * kStep2Block + lane + v * 32;
          const float wv =
              fp8_e4m3fn_to_float_device_step2(w2_e[static_cast<int64_t>(h) * kStep2Intermediate + i]);
          raw4[v] += cv * wv;
        }
      }
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        acc4[v] += raw4[v] * scale;
      }
    }

#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int h = hb * kStep2Block + lane + v * 32;
      atomicAdd(&out_acc_dev[static_cast<int64_t>(tok) * kStep2Hidden + h], tok_w * acc4[v]);
    }
  }
}

static inline cudaError_t LaunchStep2DirectAllExperts(const float* c_perm_all_dev,
                                                      const int* expert_t_valid,
                                                      const int* expert_offset,
                                                      const int* valid_token_idx,
                                                      const float* valid_token_w,
                                                      const uint8_t* w2_all_dev,
                                                      const float* s2_all_dev,
                                                      float* out_acc_dev,
                                                      cudaStream_t stream) {
  dim3 grid(kStep2Hidden / kStep2Block, kStep2LocalExperts);
  dim3 threads(32);
  step2_gemm2_scatter_all_experts_direct_kernel<<<grid, threads, 0, stream>>>(
      c_perm_all_dev, expert_t_valid, expert_offset, valid_token_idx, valid_token_w, w2_all_dev,
      s2_all_dev, out_acc_dev);
  return cudaGetLastError();
}

}  // namespace mxfp::b200::direct
