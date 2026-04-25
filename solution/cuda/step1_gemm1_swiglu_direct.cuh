#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace direct_backend {

static constexpr int kStep1Hidden = 7168;
static constexpr int kStep1Intermediate = 2048;
static constexpr int kStep1Gemm1Out = 2 * kStep1Intermediate;          // 4096
static constexpr int kStep1Block = 128;                                // outer output tile and outer K tile
static constexpr int kStep1TcgenK = 32;                                // tcgen05 inner K tile
static constexpr int kStep1RowTile = 8;                                // tcgen05 N tile in this design
static constexpr int kStep1LocalExperts = 32;
static constexpr int kStep1Threads = 32;                               // one warp
static constexpr int kStep1HiddenBlocks = kStep1Hidden / kStep1Block;  // 56
static constexpr int kStep1IntermediateBlocks = kStep1Intermediate / kStep1Block;  // 16
static constexpr int kStep1Gemm1OutBlocks = kStep1Gemm1Out / kStep1Block;          // 32
static constexpr int kStep1KSubsPerBlock = kStep1Block / kStep1TcgenK;             // 4

static_assert(kStep1Hidden % kStep1Block == 0, "Hidden size must be divisible by 128.");
static_assert(kStep1Block % kStep1TcgenK == 0, "128 must be divisible by tcgen05 K tile.");

__host__ __device__ __forceinline__ std::size_t align_up(std::size_t x, std::size_t a) {
  return (x + a - 1) & ~(a - 1);
}

__device__ __forceinline__ float fp8_e4m3fn_to_float_device(uint8_t x) {
  const int sign = (x & 0x80) ? -1 : 1;
  const int exp = (x >> 3) & 0x0f;
  const int mant = x & 0x07;

  if (exp == 0) {
    if (mant == 0) {
      return sign > 0 ? 0.0f : -0.0f;
    }
    const float frac = static_cast<float>(mant) * 0.125f;
    return sign * ldexpf(frac, -6);
  }

  // Conservatively map the canonical NaN code.
  if (exp == 0x0f && mant == 0x07) {
    return __int_as_float(0x7fffffff);
  }

  const float frac = 1.0f + static_cast<float>(mant) * 0.125f;
  return sign * ldexpf(frac, exp - 7);
}

__device__ __forceinline__ float silu_device(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ uint32_t smem_ptr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// -----------------------------------------------------------------------------
// tcgen05 / TMEM helper wrappers that are stable and compile-safe.
//
// These helpers are kept because they are useful for the future PTX path, but
// the current kernel below intentionally uses the direct fallback compute path.
// The exact tcgen05.mma/ld/st sequence is not wired here yet.
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t tcgen05_alloc_cta_group_1(uint32_t* smem_addr_slot, uint32_t cols) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  const uint32_t smem_addr = smem_ptr(smem_addr_slot);
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
      :
      : "r"(smem_addr), "r"(cols)
      : "memory");
  uint32_t taddr = 0;
  asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(taddr) : "r"(smem_addr) : "memory");
  return taddr;
#else
  (void)smem_addr_slot;
  (void)cols;
  return 0;
#endif
}

__device__ __forceinline__ void tcgen05_dealloc_cta_group_1(uint32_t taddr, uint32_t cols) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
      :
      : "r"(taddr), "r"(cols)
      : "memory");
#else
  (void)taddr;
  (void)cols;
#endif
}

__device__ __forceinline__ void tcgen05_relinquish_alloc_permit_cta_group_1() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen05_commit_arrive_one_cta_group_1(uint64_t const* smem_barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  const uint32_t bar_intptr = smem_ptr(smem_barrier);
  if ((threadIdx.x & 31) == 0) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
        :
        : "r"(bar_intptr)
        : "memory");
  }
#else
  (void)smem_barrier;
#endif
}

__device__ __forceinline__ void tcgen05_wait_ld() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen05_wait_st() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void fence_view_async_shared_cta() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}

// Shared-memory layout for one CTA:
//   [ A_gate_full | A_up_full | B_full ]
//   A_gate_full : [128][128] row-major
//   A_up_full   : [128][128] row-major
//   B_full      : [128][8]   row-major
__host__ __device__ __forceinline__ std::size_t step1_smem_bytes() {
  std::size_t offset = 0;
  offset += static_cast<std::size_t>(kStep1Block) * kStep1Block;   // A_gate_full
  offset += static_cast<std::size_t>(kStep1Block) * kStep1Block;   // A_up_full
  offset += static_cast<std::size_t>(kStep1Block) * kStep1RowTile; // B_full
  return align_up(offset, 16);
}

// -----------------------------------------------------------------------------
// Direct staging helpers.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void load_weight_full_direct(
    uint8_t* smem_A_gate_full,
    uint8_t* smem_A_up_full,
    const uint8_t* __restrict__ w13_all_dev,
    int expert,
    int n_tile,
    int k_blk) {
  const int gate_tile = n_tile;
  const int up_tile = n_tile + kStep1IntermediateBlocks;
  const int k0 = k_blk * kStep1Block;

  for (int linear = threadIdx.x; linear < kStep1Block * kStep1Block; linear += blockDim.x) {
    const int out_local = linear / kStep1Block;  // 0..127
    const int kk = linear % kStep1Block;         // 0..127
    const int k_idx = k0 + kk;
    const int gate_out = gate_tile * kStep1Block + out_local;
    const int up_out = up_tile * kStep1Block + out_local;

    smem_A_gate_full[out_local * kStep1Block + kk] =
        w13_all_dev[((expert * kStep1Gemm1Out + gate_out) * kStep1Hidden) + k_idx];

    smem_A_up_full[out_local * kStep1Block + kk] =
        w13_all_dev[((expert * kStep1Gemm1Out + up_out) * kStep1Hidden) + k_idx];
  }
}

__device__ __forceinline__ void load_hidden_full_direct(
    uint8_t* smem_B_full,
    const uint8_t* __restrict__ hidden_fp8_dev,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int row_tile,
    int n_rows,
    int k_blk) {
  const int k0 = k_blk * kStep1Block;

  for (int linear = threadIdx.x; linear < kStep1Block * kStep1RowTile; linear += blockDim.x) {
    const int kk = linear / kStep1RowTile;  // 0..127
    const int rr = linear % kStep1RowTile;  // 0..7

    uint8_t val = 0;
    if (rr < n_rows) {
      const int packed_idx = packed_base + row_tile + rr;
      const int token_idx = valid_token_idx[packed_idx];
      val = hidden_fp8_dev[token_idx * kStep1Hidden + (k0 + kk)];
    }
    smem_B_full[kk * kStep1RowTile + rr] = val;
  }
}

struct LaneAccum8x4 {
  float v[kStep1RowTile][4];

  __device__ __forceinline__ void clear() {
#pragma unroll
    for (int r = 0; r < kStep1RowTile; ++r) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        v[r][c] = 0.0f;
      }
    }
  }
};

// -----------------------------------------------------------------------------
// Compile-safe fallback compute for a 128x8x32 TT microkernel.
//
// Shared-memory physical layout:
//   A_sub: [128][32] row-major
//   B_sub: [32][8]   row-major
// Logical GEMM:
//   A(128x32) * B(32x8) = C(128x8)
//
// Lane ownership in this fallback path:
//   lane owns columns lane + 32 * v, for v in [0, 3]
//   and all 8 rows of the current row tile.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void mma_128x8x32_tt_fallback(
    LaneAccum8x4& acc,
    const uint8_t* __restrict__ smem_A_sub,
    const uint8_t* __restrict__ smem_B_sub,
    float weight_block_scale,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int row_tile,
    int n_rows,
    int k_blk,
    int lane) {
#pragma unroll
  for (int rr = 0; rr < kStep1RowTile; ++rr) {
    if (rr >= n_rows) {
      break;
    }

    const int packed_idx = packed_base + row_tile + rr;
    const int token_idx = valid_token_idx[packed_idx];
    const float hidden_block_scale = hidden_scale_dev[static_cast<int64_t>(k_blk) * t + token_idx];
    const float fused_scale = weight_block_scale * hidden_block_scale;

#pragma unroll
    for (int vv = 0; vv < 4; ++vv) {
      const int out_local = lane + vv * 32;
      float partial = 0.0f;
#pragma unroll
      for (int kk = 0; kk < kStep1TcgenK; ++kk) {
        const float a = fp8_e4m3fn_to_float_device(smem_A_sub[out_local * kStep1TcgenK + kk]);
        const float b = fp8_e4m3fn_to_float_device(smem_B_sub[kk * kStep1RowTile + rr]);
        partial += a * b;
      }
      acc.v[rr][vv] += partial * fused_scale;
    }
  }
}

// -----------------------------------------------------------------------------
// PTX hook point.
//
// The current file intentionally keeps the kernel compile-safe and runnable with
// the fallback compute path above. The exact tcgen05.mma / tcgen05.ld sequence
// is not wired yet because that requires a confirmed TMEM layout D mapping and
// exact operand descriptor encoding for the chosen f8f6f4 TT variant.
// -----------------------------------------------------------------------------
#if defined(MXFP_ENABLE_TCGEN05_PTX)
#error "MXFP_ENABLE_TCGEN05_PTX is reserved for the future inline-PTX tcgen05 path and is not wired yet."
#endif

static __global__ void step1_gemm1_swiglu_direct_kernel(
    const uint8_t* __restrict__ hidden_fp8_dev,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ expert_t_valid,
    const int* __restrict__ expert_offset,
    const int* __restrict__ valid_token_idx,
    const uint8_t* __restrict__ w13_all_dev,
    const float* __restrict__ s13_all_dev,
    const void* __restrict__ hidden_tma_desc,
    const void* __restrict__ w13_tma_desc,
    float* __restrict__ c_perm_all_dev) {
  (void)hidden_tma_desc;
  (void)w13_tma_desc;

  const int n_tile = blockIdx.x;  // 0..15
  const int expert = blockIdx.y;  // 0..31
  const int lane = threadIdx.x;

  if (lane >= kStep1Threads) {
    return;
  }
  if (expert >= kStep1LocalExperts) {
    return;
  }
  if (n_tile >= kStep1IntermediateBlocks) {
    return;
  }

  const int t_valid = expert_t_valid[expert];
  if (t_valid <= 0) {
    return;
  }

  const int packed_base = expert_offset[expert];

  extern __shared__ __align__(16) uint8_t smem_raw[];
  std::size_t offset = 0;

  uint8_t* smem_A_gate_full = smem_raw + offset;
  offset += static_cast<std::size_t>(kStep1Block) * kStep1Block;

  uint8_t* smem_A_up_full = smem_raw + offset;
  offset += static_cast<std::size_t>(kStep1Block) * kStep1Block;

  uint8_t* smem_B_full = smem_raw + offset;

  for (int row_tile = 0; row_tile < t_valid; row_tile += kStep1RowTile) {
    const int n_rows = ((t_valid - row_tile) < kStep1RowTile) ? (t_valid - row_tile) : kStep1RowTile;

    LaneAccum8x4 gate_acc;
    LaneAccum8x4 up_acc;
    gate_acc.clear();
    up_acc.clear();

    for (int k_blk = 0; k_blk < kStep1HiddenBlocks; ++k_blk) {
      load_weight_full_direct(smem_A_gate_full, smem_A_up_full, w13_all_dev, expert, n_tile, k_blk);
      load_hidden_full_direct(smem_B_full, hidden_fp8_dev, valid_token_idx, packed_base, row_tile, n_rows, k_blk);
      __syncthreads();

      const float gate_scale =
          s13_all_dev[(expert * kStep1Gemm1OutBlocks + n_tile) * kStep1HiddenBlocks + k_blk];
      const float up_scale =
          s13_all_dev[(expert * kStep1Gemm1OutBlocks + (n_tile + kStep1IntermediateBlocks)) *
                          kStep1HiddenBlocks +
                      k_blk];

#pragma unroll
      for (int k_sub = 0; k_sub < kStep1KSubsPerBlock; ++k_sub) {
        const uint8_t* smem_A_gate_sub = smem_A_gate_full + k_sub * kStep1TcgenK;
        const uint8_t* smem_A_up_sub = smem_A_up_full + k_sub * kStep1TcgenK;
        const uint8_t* smem_B_sub = smem_B_full + k_sub * kStep1TcgenK * kStep1RowTile;

        // Future tcgen05 path:
        //   tcgen05.mma.cta_group::1.kind::f8f6f4 ... TT ... on gate
        //   tcgen05.mma.cta_group::1.kind::f8f6f4 ... TT ... on up
        //   tcgen05.ld / fragment extraction
        // For now, keep a compile-safe fallback.
        mma_128x8x32_tt_fallback(
            gate_acc,
            smem_A_gate_sub,
            smem_B_sub,
            gate_scale,
            hidden_scale_dev,
            t,
            valid_token_idx,
            packed_base,
            row_tile,
            n_rows,
            k_blk,
            lane);

        mma_128x8x32_tt_fallback(
            up_acc,
            smem_A_up_sub,
            smem_B_sub,
            up_scale,
            hidden_scale_dev,
            t,
            valid_token_idx,
            packed_base,
            row_tile,
            n_rows,
            k_blk,
            lane);
      }

      __syncthreads();
    }

#pragma unroll
    for (int rr = 0; rr < kStep1RowTile; ++rr) {
      if (rr >= n_rows) {
        break;
      }
      const int packed_row = packed_base + row_tile + rr;
#pragma unroll
      for (int vv = 0; vv < 4; ++vv) {
        const int out_local = lane + vv * 32;
        const int out_col = n_tile * kStep1Block + out_local;
        const float g = gate_acc.v[rr][vv];
        const float u = up_acc.v[rr][vv];
        c_perm_all_dev[static_cast<int64_t>(packed_row) * kStep1Intermediate + out_col] =
            silu_device(g) * u;
      }
    }
  }
}

static inline cudaError_t RunStep1AllExpertsDirect(
    const uint8_t* hidden_fp8_dev,
    const float* hidden_scale_dev,
    int64_t t,
    const int* expert_counts_dev,
    const int* expert_offsets_dev,
    const int* permuted_token_ids_dev,
    const uint8_t* w13_all_dev,
    const float* s13_all_dev,
    float* c_perm_all_dev,
    cudaStream_t stream) {
  const std::size_t smem_bytes = step1_smem_bytes();

  cudaError_t st = cudaFuncSetAttribute(
      step1_gemm1_swiglu_direct_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  if (st != cudaSuccess) {
    return st;
  }

  dim3 grid(kStep1IntermediateBlocks, kStep1LocalExperts, 1);
  dim3 block(kStep1Threads, 1, 1);

  step1_gemm1_swiglu_direct_kernel<<<grid, block, smem_bytes, stream>>>(
      hidden_fp8_dev,
      hidden_scale_dev,
      t,
      expert_counts_dev,
      expert_offsets_dev,
      permuted_token_ids_dev,
      w13_all_dev,
      s13_all_dev,
      nullptr,
      nullptr,
      c_perm_all_dev);

  return cudaGetLastError();
}

}  // namespace direct_backend
