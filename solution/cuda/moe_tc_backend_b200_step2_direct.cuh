#pragma once

#include <cuda_runtime.h>
#include <cuda/__ptx/instructions/tcgen05_alloc.h>

#include <cmath>
#include <cstdint>

namespace mxfp::b200::direct {

static constexpr int kStep2Hidden = 7168;
static constexpr int kStep2Intermediate = 2048;
static constexpr int kStep2Block = 128;
static constexpr int kStep2LocalExperts = 32;
static constexpr int kStep2TmaWaitIters = 8192;

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

__device__ __forceinline__ uint32_t SmemPtrStep2(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ bool TryTmaLoad128x1Step2(void* smem_dst, const void* tensor_map,
                                                      int32_t coord_x, int32_t coord_y,
                                                      uint64_t* smem_barrier) {
  if (threadIdx.x == 0) {
    uint32_t dst_ptr = SmemPtrStep2(smem_dst);
    uint32_t bar_ptr = SmemPtrStep2(smem_barrier);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "r"(bar_ptr), "r"(1) : "memory");
    asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                 : : "r"(bar_ptr), "r"(kStep2Block) : "memory");
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
    uint32_t bar_ptr = SmemPtrStep2(smem_barrier);
    for (int i = 0; i < kStep2TmaWaitIters; ++i) {
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

// Step2 direct path for a single expert slice (legacy path kept for call-site compatibility).
__global__ void step2_gemm2_scatter_direct_kernel(const float* __restrict__ c_perm_dev, int n_rows,
                                                  const int* __restrict__ permuted_tok_e,
                                                  const float* __restrict__ permuted_w_e,
                                                  const uint8_t* __restrict__ w2_e,
                                                  const float* __restrict__ s2_e,
                                                  float* __restrict__ out_acc_dev) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * kStep2Hidden;
  if (idx >= total) return;

  int row = static_cast<int>(idx / kStep2Hidden);
  int h = static_cast<int>(idx - static_cast<int64_t>(row) * kStep2Hidden);
  int tok = permuted_tok_e[row];
  float token_w = permuted_w_e[row];
  int hb = h / kStep2Block;
  int intermediate_blocks = kStep2Intermediate / kStep2Block;

  const float* c_row = c_perm_dev + static_cast<int64_t>(row) * kStep2Intermediate;
  const uint8_t* w_row = w2_e + static_cast<int64_t>(h) * kStep2Intermediate;
  float acc = 0.0f;
  for (int ib = 0; ib < intermediate_blocks; ++ib) {
    float scale = s2_e[hb * intermediate_blocks + ib];
    int i0 = ib * kStep2Block;
    float raw = 0.0f;
    for (int u = 0; u < kStep2Block; ++u) {
      raw += c_row[i0 + u] * fp8_e4m3fn_to_float_device_step2(w_row[i0 + u]);
    }
    acc += raw * scale;
  }

  atomicAdd(&out_acc_dev[static_cast<int64_t>(tok) * kStep2Hidden + h], token_w * acc);
}

static inline cudaError_t LaunchStep2Direct(const float* c_perm_dev, int n_rows,
                                            const int* permuted_tok_e,
                                            const float* permuted_w_e, const uint8_t* w2_e,
                                            const float* s2_e, float* out_acc_dev,
                                            cudaStream_t stream) {
  if (n_rows <= 0) return cudaSuccess;
  constexpr int kThreads = 128;
  int64_t n_out = static_cast<int64_t>(n_rows) * kStep2Hidden;
  int blocks = static_cast<int>((n_out + kThreads - 1) / kThreads);
  step2_gemm2_scatter_direct_kernel<<<blocks, kThreads, 0, stream>>>(
      c_perm_dev, n_rows, permuted_tok_e, permuted_w_e, w2_e, s2_e, out_acc_dev);
  return cudaGetLastError();
}

// Step2 all-experts direct path:
// - grid.x = H/128 (56 tiles), grid.y = 32 local experts
// - blockDim.x = 32, each lane computes 4 output channels
// - each CTA iterates all routed rows for its expert
// - writes to out_acc use atomicAdd because experts run concurrently and may
//   update the same (tok, h) location
// - TMA is used for weight 128-element K slices with bounded wait + fallback
// - tcgen05 TMEM allocator is exercised per CTA (alloc/relinquish/dealloc)
__global__ void step2_gemm2_scatter_all_experts_direct_kernel(
    const float* __restrict__ c_perm_all_dev, const int* __restrict__ expert_t_valid,
    const int* __restrict__ expert_offset, const int* __restrict__ valid_token_idx,
    const float* __restrict__ valid_token_w, const uint8_t* __restrict__ w2_all_dev,
    const float* __restrict__ s2_all_dev, const void* __restrict__ w2_tma_desc,
    float* __restrict__ out_acc_dev) {
  const int hb = blockIdx.x;
  const int expert = blockIdx.y;
  const int lane = threadIdx.x;
  if (lane >= 32) return;
  if (expert >= kStep2LocalExperts) return;
  if (hb >= (kStep2Hidden / kStep2Block)) return;

  __shared__ uint32_t tmem_base;
  cuda::ptx::tcgen05_alloc(cuda::ptx::cta_group_1, &tmem_base, uint32_t(32));
  cuda::ptx::tcgen05_relinquish_alloc_permit(cuda::ptx::cta_group_1);
  __syncwarp();

  const int t_valid = expert_t_valid[expert];
  if (t_valid <= 0) {
    cuda::ptx::tcgen05_dealloc(cuda::ptx::cta_group_1, tmem_base, uint32_t(32));
    return;
  }
  const int row_start = expert_offset[expert];

  const int intermediate_blocks = kStep2Intermediate / kStep2Block;
  const size_t w2_expert_elems = static_cast<size_t>(kStep2Hidden) * kStep2Intermediate;
  const size_t s2_expert_elems = static_cast<size_t>(kStep2Hidden / kStep2Block) * intermediate_blocks;
  const uint8_t* w2_e = w2_all_dev + static_cast<size_t>(expert) * w2_expert_elems;
  const float* s2_e = s2_all_dev + static_cast<size_t>(expert) * s2_expert_elems;

  alignas(16) __shared__ uint8_t s_w[4][kStep2Block];
  alignas(16) __shared__ uint64_t bar_w[4];

  for (int row_local = 0; row_local < t_valid; ++row_local) {
    const int slot = row_start + row_local;
    const int tok = valid_token_idx[slot];
    const float tok_w = valid_token_w[slot];
    const float* c_row = c_perm_all_dev + static_cast<int64_t>(slot) * kStep2Intermediate;

    float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int ib = 0; ib < intermediate_blocks; ++ib) {
      const float scale = s2_e[hb * intermediate_blocks + ib];
      const int i0 = ib * kStep2Block;
      float raw4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      bool tma_w_ok[4];
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int h = hb * kStep2Block + lane + v * 32;
        const int w_row_global = expert * kStep2Hidden + h;
        tma_w_ok[v] = TryTmaLoad128x1Step2(s_w[v], w2_tma_desc, i0, w_row_global, &bar_w[v]);
        if (!tma_w_ok[v]) {
          for (int uu = lane; uu < kStep2Block; uu += 32) {
            s_w[v][uu] = w2_e[static_cast<int64_t>(h) * kStep2Intermediate + (i0 + uu)];
          }
        }
      }
      __syncwarp();

      for (int u = 0; u < kStep2Block; ++u) {
        const int i = i0 + u;
        const float cv = c_row[i];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const float wv = fp8_e4m3fn_to_float_device_step2(s_w[v][u]);
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

  __syncwarp();
  cuda::ptx::tcgen05_dealloc(cuda::ptx::cta_group_1, tmem_base, uint32_t(32));
}

static inline cudaError_t LaunchStep2DirectAllExperts(
    const float* c_perm_all_dev, const int* expert_t_valid, const int* expert_offset,
    const int* valid_token_idx, const float* valid_token_w, const uint8_t* w2_all_dev,
    const float* s2_all_dev, const void* w2_tma_desc, float* out_acc_dev, cudaStream_t stream) {
  dim3 grid(kStep2Hidden / kStep2Block, kStep2LocalExperts);
  dim3 threads(32);
  step2_gemm2_scatter_all_experts_direct_kernel<<<grid, threads, 0, stream>>>(
      c_perm_all_dev, expert_t_valid, expert_offset, valid_token_idx, valid_token_w, w2_all_dev,
      s2_all_dev, w2_tma_desc, out_acc_dev);
  return cudaGetLastError();
}

}  // namespace mxfp::b200::direct
