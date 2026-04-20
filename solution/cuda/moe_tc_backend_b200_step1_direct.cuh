#pragma once

#include "b200_tcgen05_ptx.cuh"
#include "b200_tma_ptx.cuh"
#include "b200_tmem_ptx.cuh"

#include <cuda.h>
#include <cuda/ptx>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace mxfp::b200::direct {

static constexpr int kStep1Hidden = 7168;
static constexpr int kStep1Intermediate = 2048;
static constexpr int kStep1Block = 128;
static constexpr int kStep1LocalExperts = 32;

__device__ __forceinline__ float fp8_e4m3fn_to_float_device(uint8_t x) {
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

__device__ __forceinline__ float siluf_device(float x) { return x / (1.0f + expf(-x)); }

// Step1 direct path with expert-parallel fixed tiling:
// - grid.x = I/128 (16 tiles), grid.y = 32 local experts
// - blockDim.x = 32 (one warp), each lane computes 4 output columns
// - for each expert and i-tile, iterate rows in that expert's routed segment
// - K is processed as 56 blocks of 128
// - activation/weight scales remain FP32 and are applied after block partial sums
//
// Input routing metadata:
// - expert_t_valid[e]: number of routed tokens for expert e
// - expert_offset[e]: starting index in valid_token_idx for expert e
// - valid_token_idx[offset[e] + r]: original token id for row r of expert e
//
// Output layout:
// - c_perm_all_dev is packed by expert segments using the same global slot index
//   (slot = expert_offset[e] + r), shape [num_routed, I].
__global__ void step1_gemm1_swiglu_direct_kernel(const uint8_t* __restrict__ hidden_fp8_dev,
                                                 const float* __restrict__ hidden_scale_dev,
                                                 int64_t t,
                                                 const int* __restrict__ expert_t_valid,
                                                 const int* __restrict__ expert_offset,
                                                 const int* __restrict__ valid_token_idx,
                                                 const uint8_t* __restrict__ w13_all_dev,
                                                 const float* __restrict__ s13_all_dev,
                                                 const CUtensorMap* __restrict__ hidden_tmap_dev,
                                                 const CUtensorMap* __restrict__ w13_tmap_dev,
                                                 float* __restrict__ c_perm_all_dev) {
  const int itile = blockIdx.x;
  const int expert = blockIdx.y;
  const int lane = threadIdx.x;
  if (lane >= 32) return;
  if (expert >= kStep1LocalExperts) return;
  if (itile >= (kStep1Intermediate / kStep1Block)) return;

  const int hidden_blocks = kStep1Hidden / kStep1Block;
  const int gemm1_out_blocks = (2 * kStep1Intermediate) / kStep1Block;
  const int expert_i_offset = itile * kStep1Block;
  const int t_valid = expert_t_valid[expert];
  if (t_valid <= 0) return;
  const int row_start = expert_offset[expert];

  const size_t w13_expert_elems = static_cast<size_t>(2 * kStep1Intermediate) * kStep1Hidden;
  const size_t s13_expert_elems =
      static_cast<size_t>(gemm1_out_blocks) * hidden_blocks;
  const uint8_t* w13_e = w13_all_dev + static_cast<size_t>(expert) * w13_expert_elems;
  const float* s13_e = s13_all_dev + static_cast<size_t>(expert) * s13_expert_elems;

  const int gate_tile = itile;
  const int up_tile = itile + (kStep1Intermediate / kStep1Block);
  const bool use_tma = (hidden_tmap_dev != nullptr && w13_tmap_dev != nullptr);

  __shared__ float s_out_tile[kStep1Block];
  __shared__ alignas(16) uint8_t s_a_tile[kStep1Block];
  __shared__ alignas(16) uint8_t s_w_gate_tile[kStep1Block * kStep1Block];
  __shared__ alignas(16) uint8_t s_w_up_tile[kStep1Block * kStep1Block];
  __shared__ alignas(8) uint64_t s_tma_bar;

  for (int row_local = 0; row_local < t_valid; ++row_local) {
    const int slot = row_start + row_local;
    const int tok = valid_token_idx[slot];
    const uint8_t* a_row = hidden_fp8_dev + static_cast<int64_t>(tok) * kStep1Hidden;

    float acc_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int hb = 0; hb < hidden_blocks; ++hb) {
      const int h0 = hb * kStep1Block;
      if (lane == 0) {
        if (use_tma) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
          cuda::ptx::mbarrier_init(&s_tma_bar, 3);
#endif
          ptx::TmaDesc a_tma_desc{};
          a_tma_desc.tensor_map = hidden_tmap_dev;
          a_tma_desc.smem_bar = &s_tma_bar;
          a_tma_desc.cta_group = 1;
          a_tma_desc.valid = true;
          ptx::tma_async_load_2d(a_tma_desc, s_a_tile, tok, h0);

          ptx::TmaDesc w_tma_desc{};
          w_tma_desc.tensor_map = w13_tmap_dev;
          w_tma_desc.smem_bar = &s_tma_bar;
          w_tma_desc.cta_group = 1;
          w_tma_desc.valid = true;
          ptx::tma_async_load_2d(w_tma_desc, s_w_gate_tile, expert_i_offset, h0);
          ptx::tma_async_load_2d(w_tma_desc, s_w_up_tile, expert_i_offset + kStep1Intermediate, h0);
          ptx::tma_commit_group();
          ptx::tma_wait_group(0);
        }

        auto tmem_h = ptx::tmem_alloc(1);
        ptx::Tcgen05MmaDesc mma_desc{};
        ptx::tcgen05_mma_f8(mma_desc);
        ptx::tmem_wait();
        ptx::tmem_cp();
        ptx::tmem_dealloc(tmem_h);
      }
      __syncthreads();

      const float a_scale = hidden_scale_dev[static_cast<int64_t>(hb) * t + tok];
      const float scale_gate = s13_e[gate_tile * hidden_blocks + hb];
      const float scale_up = s13_e[up_tile * hidden_blocks + hb];

      float raw_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float raw_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int u = 0; u < kStep1Block; ++u) {
        const int h = h0 + u;
        const float a = fp8_e4m3fn_to_float_device(use_tma ? s_a_tile[u] : a_row[h]) * a_scale;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const int col = lane + v * 32;
          const int j_gate = expert_i_offset + col;
          const int j_up = j_gate + kStep1Intermediate;
          const float wg = fp8_e4m3fn_to_float_device(
              use_tma ? s_w_gate_tile[col * kStep1Block + u]
                      : w13_e[static_cast<int64_t>(j_gate) * kStep1Hidden + h]);
          const float wu = fp8_e4m3fn_to_float_device(
              use_tma ? s_w_up_tile[col * kStep1Block + u]
                      : w13_e[static_cast<int64_t>(j_up) * kStep1Hidden + h]);
          raw_gate[v] += a * wg;
          raw_up[v] += a * wu;
        }
      }

#pragma unroll
      for (int v = 0; v < 4; ++v) {
        acc_gate[v] += raw_gate[v] * scale_gate;
        acc_up[v] += raw_up[v] * scale_up;
      }
      __syncthreads();
    }
    // Stage per-row tile result in shared memory before final global write.
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int col = lane + v * 32;
      s_out_tile[col] = acc_gate[v] * siluf_device(acc_up[v]);
    }
    __syncthreads();

    if (lane == 0) {
      // TODO(B200): Replace with real TMA store from staged tile to global.
      ptx::TmaDesc tma_store_desc{};
      ptx::tma_async_load_2d(tma_store_desc, nullptr, itile, slot);
      ptx::tma_commit_group();
      ptx::tma_wait_group(0);
    }

#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int col = lane + v * 32;
      c_perm_all_dev[static_cast<int64_t>(slot) * kStep1Intermediate + expert_i_offset + col] =
          s_out_tile[col];
    }
    __syncthreads();
  }
}

static inline cudaError_t LaunchStep1DirectAllExperts(const uint8_t* hidden_fp8_dev,
                                                      const float* hidden_scale_dev,
                                                      int64_t t,
                                                      const int* expert_t_valid,
                                                      const int* expert_offset,
                                                      const int* valid_token_idx,
                                                      const uint8_t* w13_all_dev,
                                                      const float* s13_all_dev,
                                                      const CUtensorMap* hidden_tmap_dev,
                                                      const CUtensorMap* w13_tmap_dev,
                                                      float* c_perm_all_dev,
                                                      cudaStream_t stream) {
  dim3 grid(kStep1Intermediate / kStep1Block, kStep1LocalExperts);
  dim3 threads(32);
  step1_gemm1_swiglu_direct_kernel<<<grid, threads, 0, stream>>>(
      hidden_fp8_dev, hidden_scale_dev, t, expert_t_valid, expert_offset, valid_token_idx,
      w13_all_dev, s13_all_dev, hidden_tmap_dev, w13_tmap_dev, c_perm_all_dev);
  return cudaGetLastError();
}

}  // namespace mxfp::b200::direct
