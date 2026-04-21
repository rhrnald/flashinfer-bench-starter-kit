#pragma once

#include <cuda_runtime.h>
#include <cuda/__ptx/instructions/tcgen05_alloc.h>

#include <cmath>
#include <cstdint>

namespace mxfp::b200::direct {

static constexpr int kStep1Hidden = 7168;
static constexpr int kStep1Intermediate = 2048;
static constexpr int kStep1Block = 128;
static constexpr int kStep1LocalExperts = 32;
static constexpr int kStep1TmaWaitIters = 8192;

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

__device__ __forceinline__ uint32_t SmemPtr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ bool TryTmaLoad128x1(void* smem_dst, const void* tensor_map,
                                                 int32_t coord_x, int32_t coord_y,
                                                 uint64_t* smem_barrier) {
  if (threadIdx.x == 0) {
    uint32_t dst_ptr = SmemPtr(smem_dst);
    uint32_t bar_ptr = SmemPtr(smem_barrier);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "r"(bar_ptr), "r"(1) : "memory");
    asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                 : : "r"(bar_ptr), "r"(kStep1Block) : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 "
        "[%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(dst_ptr), "l"(tensor_map), "r"(coord_x), "r"(coord_y), "r"(bar_ptr)
        : "memory");
  }
  __syncwarp();

  int ready = 0;
  if (threadIdx.x == 0) {
    uint32_t bar_ptr = SmemPtr(smem_barrier);
    for (int i = 0; i < kStep1TmaWaitIters; ++i) {
      int r = 0;
      asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
          "selp.b32 %0, 1, 0, p;\n\t"
          "}\n"
          : "=r"(r)
          : "r"(bar_ptr), "r"(0));
      if (r) {
        ready = 1;
        break;
      }
    }
  }
  ready = __shfl_sync(0xffffffffu, ready, 0);
  return ready != 0;
}

// Step1 direct path with expert-parallel fixed tiling:
// - grid.x = I/128 (16 tiles), grid.y = 32 local experts
// - blockDim.x = 32 (one warp), each lane computes 4 output columns
// - for each expert and i-tile, iterate rows in that expert's routed segment
// - K is processed as 56 blocks of 128
// - activation/weight scales remain FP32 and are applied after block partial sums
// - TMA is used for loading A/B 128-element K slices with bounded wait + fallback
// - tcgen05 TMEM allocator is exercised per CTA (alloc/relinquish/dealloc)
__global__ void step1_gemm1_swiglu_direct_kernel(
    const uint8_t* __restrict__ hidden_fp8_dev, const float* __restrict__ hidden_scale_dev,
    int64_t t, const int* __restrict__ expert_t_valid, const int* __restrict__ expert_offset,
    const int* __restrict__ valid_token_idx, const uint8_t* __restrict__ w13_all_dev,
    const float* __restrict__ s13_all_dev, const void* __restrict__ hidden_tma_desc,
    const void* __restrict__ w13_tma_desc, float* __restrict__ c_perm_all_dev) {
  const int itile = blockIdx.x;
  const int expert = blockIdx.y;
  const int lane = threadIdx.x;
  if (lane >= 32) return;
  if (expert >= kStep1LocalExperts) return;
  if (itile >= (kStep1Intermediate / kStep1Block)) return;

  __shared__ uint32_t tmem_base;
  cuda::ptx::tcgen05_alloc(cuda::ptx::cta_group_1, &tmem_base, uint32_t(32));
  cuda::ptx::tcgen05_relinquish_alloc_permit(cuda::ptx::cta_group_1);
  __syncwarp();

  const int hidden_blocks = kStep1Hidden / kStep1Block;
  const int gemm1_out_blocks = (2 * kStep1Intermediate) / kStep1Block;
  const int expert_i_offset = itile * kStep1Block;
  const int t_valid = expert_t_valid[expert];
  if (t_valid <= 0) {
    cuda::ptx::tcgen05_dealloc(cuda::ptx::cta_group_1, tmem_base, uint32_t(32));
    return;
  }
  const int row_start = expert_offset[expert];

  const size_t w13_expert_elems = static_cast<size_t>(2 * kStep1Intermediate) * kStep1Hidden;
  const size_t s13_expert_elems = static_cast<size_t>(gemm1_out_blocks) * hidden_blocks;
  const uint8_t* w13_e = w13_all_dev + static_cast<size_t>(expert) * w13_expert_elems;
  const float* s13_e = s13_all_dev + static_cast<size_t>(expert) * s13_expert_elems;

  const int gate_tile = itile;
  const int up_tile = itile + (kStep1Intermediate / kStep1Block);

  alignas(16) __shared__ uint8_t s_a[kStep1Block];
  alignas(16) __shared__ uint8_t s_w_gate[4][kStep1Block];
  alignas(16) __shared__ uint8_t s_w_up[4][kStep1Block];
  alignas(16) __shared__ uint64_t bar_a;
  alignas(16) __shared__ uint64_t bar_w;

  for (int row_local = 0; row_local < t_valid; ++row_local) {
    const int slot = row_start + row_local;
    const int tok = valid_token_idx[slot];
    const uint8_t* a_row = hidden_fp8_dev + static_cast<int64_t>(tok) * kStep1Hidden;

    float acc_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int hb = 0; hb < hidden_blocks; ++hb) {
      const int h0 = hb * kStep1Block;
      const float a_scale = hidden_scale_dev[static_cast<int64_t>(hb) * t + tok];
      const float scale_gate = s13_e[gate_tile * hidden_blocks + hb];
      const float scale_up = s13_e[up_tile * hidden_blocks + hb];

      bool tma_a_ok = TryTmaLoad128x1(s_a, hidden_tma_desc, h0, tok, &bar_a);
      bool tma_wg_ok[4];
      bool tma_wu_ok[4];
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int col = lane + v * 32;
        const int j_gate = expert_i_offset + col;
        const int j_up = j_gate + kStep1Intermediate;
        const int j_gate_global = expert * (2 * kStep1Intermediate) + j_gate;
        const int j_up_global = j_gate_global + kStep1Intermediate;
        tma_wg_ok[v] = TryTmaLoad128x1(s_w_gate[v], w13_tma_desc, h0, j_gate_global, &bar_w);
        tma_wu_ok[v] = TryTmaLoad128x1(s_w_up[v], w13_tma_desc, h0, j_up_global, &bar_w);
      }

      if (!tma_a_ok) {
        for (int u = lane; u < kStep1Block; u += 32) {
          s_a[u] = a_row[h0 + u];
        }
      }
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        if (!tma_wg_ok[v]) {
          const int col = lane + v * 32;
          const int j_gate = expert_i_offset + col;
          for (int u = lane; u < kStep1Block; u += 32) {
            s_w_gate[v][u] = w13_e[static_cast<int64_t>(j_gate) * kStep1Hidden + (h0 + u)];
          }
        }
        if (!tma_wu_ok[v]) {
          const int col = lane + v * 32;
          const int j_up = expert_i_offset + col + kStep1Intermediate;
          for (int u = lane; u < kStep1Block; u += 32) {
            s_w_up[v][u] = w13_e[static_cast<int64_t>(j_up) * kStep1Hidden + (h0 + u)];
          }
        }
      }
      __syncwarp();

      float raw_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float raw_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int u = 0; u < kStep1Block; ++u) {
        const float a = fp8_e4m3fn_to_float_device(s_a[u]) * a_scale;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const float wg = fp8_e4m3fn_to_float_device(s_w_gate[v][u]);
          const float wu = fp8_e4m3fn_to_float_device(s_w_up[v][u]);
          raw_gate[v] += a * wg;
          raw_up[v] += a * wu;
        }
      }

#pragma unroll
      for (int v = 0; v < 4; ++v) {
        acc_gate[v] += raw_gate[v] * scale_gate;
        acc_up[v] += raw_up[v] * scale_up;
      }
    }

#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int col = lane + v * 32;
      c_perm_all_dev[static_cast<int64_t>(slot) * kStep1Intermediate + expert_i_offset + col] =
          acc_gate[v] * siluf_device(acc_up[v]);
    }
  }

  __syncwarp();
  cuda::ptx::tcgen05_dealloc(cuda::ptx::cta_group_1, tmem_base, uint32_t(32));
}

static inline cudaError_t LaunchStep1DirectAllExperts(
    const uint8_t* hidden_fp8_dev, const float* hidden_scale_dev, int64_t t, const int* expert_t_valid,
    const int* expert_offset, const int* valid_token_idx, const uint8_t* w13_all_dev,
    const float* s13_all_dev, const void* hidden_tma_desc, const void* w13_tma_desc,
    float* c_perm_all_dev, cudaStream_t stream) {
  dim3 grid(kStep1Intermediate / kStep1Block, kStep1LocalExperts);
  dim3 threads(32);
  step1_gemm1_swiglu_direct_kernel<<<grid, threads, 0, stream>>>(
      hidden_fp8_dev, hidden_scale_dev, t, expert_t_valid, expert_offset, valid_token_idx,
      w13_all_dev, s13_all_dev, hidden_tma_desc, w13_tma_desc, c_perm_all_dev);
  return cudaGetLastError();
}

}  // namespace mxfp::b200::direct
