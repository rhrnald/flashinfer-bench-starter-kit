#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/ptx>
#include <cutlass/arch/barrier.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstddef>

namespace direct_backend {

#define MXFP_ENABLE_TCGEN05_PTX_ACTIVE 1
static constexpr bool kStep1UseTcgen05 = true;

static constexpr int kStep1Hidden = 7168;
static constexpr int kStep1Intermediate = 2048;
static constexpr int kStep1Gemm1Out = 2 * kStep1Intermediate;                     // 4096
static constexpr int kStep1Block = 128;                                           // outer K block
static constexpr int kStep1TcgenK = 32;                                           // tcgen05 K-subtile
static constexpr int kStep1RowTile = 8;                                           // token rows per iteration
static constexpr int kStep1OutRowsPerCta = 32;                                    // one CTA owns 32 gate rows + 32 up rows
static constexpr int kStep1LocalExperts = 32;
static constexpr int kStep1Threads = 32;                                          // one warp
static constexpr int kStep1CommThreads = 256;                                     // eight warps for cooperative staging
static constexpr int kStep1CommWarps = kStep1CommThreads / kStep1Threads;
static constexpr int kStep1HiddenBlocks = kStep1Hidden / kStep1Block;             // 56
static constexpr int kStep1TmaChunkBlocks = 4;
static constexpr int kStep1TmaChunkGroups = kStep1HiddenBlocks / kStep1TmaChunkBlocks;
static constexpr int kStep1TmaRawChunkBlocks = 8;                                 // 1024-byte K-group
static constexpr int kStep1TmaRawChunkGroups = kStep1HiddenBlocks / kStep1TmaRawChunkBlocks;  // 7
static constexpr int kStep1IntermediateBlocks = kStep1Intermediate / kStep1Block; // 16
static constexpr int kStep1Gemm1OutBlocks = kStep1Gemm1Out / kStep1Block;         // 32
static constexpr int kStep1KSubsPerBlock = kStep1Block / kStep1TcgenK;            // 4
static constexpr int kStep1OutTilesPerExpert = kStep1Intermediate / kStep1OutRowsPerCta;       // 64
static constexpr int kStep1RowsPer128Block = kStep1Block / kStep1OutRowsPerCta;                // 4
static constexpr int kStep1HiddenGroupBytes = kStep1Block * kStep1TmaRawChunkBlocks;           // 1024
static constexpr int kStep1HiddenGroupWords = kStep1HiddenGroupBytes / 4;                       // 256
static constexpr int kStep1TcgenMmaKBytes = 32;
static constexpr int kStep1TcgenSubKBytes = 64;
static constexpr int kStep1TcgenM = 64;
static constexpr int kStep1TcgenN = 8;
static constexpr int kStep1TcgenTmemCols = 32;
static constexpr int kStep1TcgenCompactKBytes = 32;
static constexpr int kStep1TcgenACompactBytes = kStep1TcgenM * kStep1TcgenCompactKBytes;       // 2048
static constexpr int kStep1TcgenBCompactBytes = kStep1TcgenN * kStep1TcgenCompactKBytes;       // 256

static_assert(kStep1Hidden % kStep1Block == 0, "Hidden size must be divisible by 128.");
static_assert(kStep1Block % kStep1TcgenK == 0, "128-block must be divisible by tcgen05 K=32.");
static_assert(kStep1CommThreads % kStep1Threads == 0, "Comm threads must be whole warps.");
static_assert(kStep1HiddenBlocks % kStep1TmaChunkBlocks == 0, "TMA chunking requires exact division.");
static_assert(kStep1HiddenBlocks % kStep1TmaRawChunkBlocks == 0, "1024-byte K grouping requires exact division.");
static_assert(kStep1Intermediate % kStep1OutRowsPerCta == 0, "Intermediate size must divide the CTA row tile.");

__host__ __device__ __forceinline__ std::size_t align_up(std::size_t x, std::size_t a) {
  return (x + a - 1) & ~(a - 1);
}

__host__ __device__ __forceinline__ int min_int(int a, int b) {
  return a < b ? a : b;
}

__device__ __forceinline__ float fp8_e4m3fn_to_float_device(uint8_t x) {
  const int sign = (x & 0x80) ? -1 : 1;
  const int exp = (x >> 3) & 0x0f;
  const int mant = x & 0x07;

  if (exp == 0) {
    if (mant == 0) return sign > 0 ? 0.0f : -0.0f;
    const float frac = static_cast<float>(mant) * 0.125f;
    return sign * ldexpf(frac, -6);
  }

  // Conservatively preserve the E4M3FN NaN encoding.
  if (exp == 0x0f && mant == 0x07) {
    return nanf("");
  }

  const float frac = 1.0f + static_cast<float>(mant) * 0.125f;
  return sign * ldexpf(frac, exp - 7);
}

__device__ __forceinline__ float silu_device(float x) {
  return x / (1.0f + expf(-x));
}

// Shared-memory layout for the restructured direct kernel:
//   [ weight_group_combined | hidden_group ]
// where weight_group_combined stores gate/up together:
//   [2][32][1024] with dim order [gate_up][row][k_byte]
// and hidden_group stores:
//   [8][1024] row-major so K stays contiguous in memory.
__host__ __device__ __forceinline__ std::size_t step1_smem_bytes() {
  const std::size_t weight_group_bytes =
      static_cast<std::size_t>(2) * kStep1OutRowsPerCta * kStep1HiddenGroupBytes;
  const std::size_t hidden_group_bytes =
      static_cast<std::size_t>(kStep1HiddenGroupBytes) * kStep1RowTile;
  std::size_t offset = 0;
  offset += weight_group_bytes;
  offset += hidden_group_bytes;
  offset = align_up(offset, 128);
  offset += kStep1TcgenACompactBytes;
  offset = align_up(offset, 128);
  offset += kStep1TcgenBCompactBytes;
  return align_up(offset, 1024);
}

__host__ __device__ __forceinline__ std::size_t step1_tma_comm_smem_bytes(
    int comm_h_inner_bytes, int comm_out_rows, int comm_h_tiles, bool comm_double_buffer) {
  (void)comm_double_buffer;
  std::size_t offset = 0;
  const std::size_t weight_group_bytes =
      static_cast<std::size_t>(2) * comm_out_rows * comm_h_inner_bytes * comm_h_tiles;
  const std::size_t hidden_group_bytes =
      static_cast<std::size_t>(comm_h_inner_bytes) * comm_h_tiles * kStep1RowTile;
  offset += weight_group_bytes;
  offset += hidden_group_bytes;
  return align_up(offset, 1024);
}

__device__ __forceinline__ uint8_t extract_byte(uint32_t word, int byte_idx) {
  return static_cast<uint8_t>((word >> (byte_idx * 8)) & 0xffu);
}

// -----------------------------------------------------------------------------
// Direct shared-memory staging helpers.
// These remain valid even after the tcgen05 inline PTX path replaces the
// fallback compute path.
// -----------------------------------------------------------------------------
template <int kThreads, typename SharedWordPtr>
__device__ __forceinline__ void load_weight_group_vec(
    SharedWordPtr smem_A_gate_words,
    SharedWordPtr smem_A_up_words,
    const uint4* __restrict__ w13_all_vec_dev,
    int expert,
    int out_row_base,
    int k_group) {
  constexpr int kVecBytes = sizeof(uint4);
  constexpr int kGroupVecsPerRow = kStep1HiddenGroupBytes / kVecBytes;   // 64
  constexpr int kHiddenVecs = kStep1Hidden / kVecBytes;                  // 448
  const int total_chunks = kStep1OutRowsPerCta * kGroupVecsPerRow;
  const int k0_vec = k_group * kGroupVecsPerRow;

  auto* smem_A_gate_vec = reinterpret_cast<uint4*>(const_cast<uint32_t*>(smem_A_gate_words));
  auto* smem_A_up_vec = reinterpret_cast<uint4*>(const_cast<uint32_t*>(smem_A_up_words));

  for (int chunk_idx = threadIdx.x; chunk_idx < total_chunks; chunk_idx += kThreads) {
    const int out_local = chunk_idx / kGroupVecsPerRow;
    const int vec_in_row = chunk_idx - out_local * kGroupVecsPerRow;
    const int gate_out = out_row_base + out_local;
    const int up_out = gate_out + kStep1Intermediate;
    const int gate_idx =
        ((expert * kStep1Gemm1Out + gate_out) * kHiddenVecs) + k0_vec + vec_in_row;
    const int up_idx =
        ((expert * kStep1Gemm1Out + up_out) * kHiddenVecs) + k0_vec + vec_in_row;
    smem_A_gate_vec[chunk_idx] = w13_all_vec_dev[gate_idx];
    smem_A_up_vec[chunk_idx] = w13_all_vec_dev[up_idx];
  }
}

template <int kThreads, typename SharedWordPtr>
__device__ __forceinline__ void load_weight_group_vec_combined(
    SharedWordPtr smem_A_group_words,
    const uint4* __restrict__ w13_all_vec_dev,
    int expert,
    int out_row_base,
    int k_group) {
  uint32_t* smem_A_gate_words = const_cast<uint32_t*>(smem_A_group_words);
  uint32_t* smem_A_up_words =
      smem_A_gate_words + (kStep1OutRowsPerCta * kStep1HiddenGroupBytes) / 4;
  load_weight_group_vec<kThreads>(
      smem_A_gate_words, smem_A_up_words, w13_all_vec_dev, expert, out_row_base, k_group);
}

template <int kThreads, typename SharedWordPtr>
__device__ __forceinline__ void load_weight_wordtiles_group_combined(
    SharedWordPtr smem_A_group_words,
    const uint4* __restrict__ w13_all_vec_dev,
    int expert,
    int out_row_base,
    int out_rows,
    int k_blk_base,
    int k_blk_count) {
  constexpr int kVecBytes = sizeof(uint4);
  constexpr int kVecsPerKBlock = kStep1Block / kVecBytes;
  constexpr int kHiddenVecs = kStep1Hidden / kVecBytes;
  const int total_chunks = out_rows * k_blk_count * kVecsPerKBlock;
  auto* smem_A_group_vec = reinterpret_cast<uint4*>(const_cast<uint32_t*>(smem_A_group_words));
  const int half_vecs = (out_rows * k_blk_count * kStep1Block) / kVecBytes;

  for (int chunk_idx = threadIdx.x; chunk_idx < total_chunks; chunk_idx += kThreads) {
    const int vecs_per_row = k_blk_count * kVecsPerKBlock;
    const int out_local = chunk_idx / vecs_per_row;
    const int row_offset = chunk_idx - out_local * vecs_per_row;
    const int blk_local = row_offset / kVecsPerKBlock;
    const int vec_in_block = row_offset - blk_local * kVecsPerKBlock;
    const int k_blk = k_blk_base + blk_local;
    const int k0_vec = (k_blk * kStep1Block) / kVecBytes;
    const int gate_out = out_row_base + out_local;
    const int up_out = gate_out + kStep1Intermediate;
    const int gate_idx =
        ((expert * kStep1Gemm1Out + gate_out) * kHiddenVecs) + k0_vec + vec_in_block;
    const int up_idx =
        ((expert * kStep1Gemm1Out + up_out) * kHiddenVecs) + k0_vec + vec_in_block;
    smem_A_group_vec[chunk_idx] = w13_all_vec_dev[gate_idx];
    smem_A_group_vec[half_vecs + chunk_idx] = w13_all_vec_dev[up_idx];
  }
}

template <int kThreads, typename SharedWordPtr>
__device__ __forceinline__ void load_hidden_group_words(
    SharedWordPtr smem_B_words,
    const uint32_t* __restrict__ hidden_words_dev,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int row_tile,
    int n_rows,
    int k_group) {
  constexpr int kHiddenWords = kStep1Hidden / 4;
  const int k0_word = k_group * kStep1HiddenGroupWords;
  const int total_words = kStep1HiddenGroupWords * n_rows;

  for (int linear_idx = threadIdx.x; linear_idx < total_words; linear_idx += kThreads) {
    const int kk_word = linear_idx / n_rows;
    const int rr = linear_idx - kk_word * n_rows;
    const int packed_idx = packed_base + row_tile + rr;
    const int token_idx = valid_token_idx[packed_idx];
    const int word_idx = static_cast<int64_t>(token_idx) * kHiddenWords + k0_word + kk_word;
    smem_B_words[rr * kStep1HiddenGroupWords + kk_word] = hidden_words_dev[word_idx];
  }
}

template <int kThreads, typename SharedWordPtr>
__device__ __forceinline__ void load_hidden_wordtiles_group(
    SharedWordPtr smem_B_words,
    const uint32_t* __restrict__ hidden_words_dev,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int row_tile,
    int n_rows,
    int k_blk_base,
    int k_blk_count) {
  constexpr int kHiddenWords = kStep1Hidden / 4;
  constexpr int kWordsPerKBlock = kStep1Block / 4;
  const int total_words = k_blk_count * kWordsPerKBlock * kStep1RowTile;

  for (int linear_idx = threadIdx.x; linear_idx < total_words; linear_idx += kThreads) {
    const int words_per_block = kWordsPerKBlock * kStep1RowTile;
    const int blk_local = linear_idx / words_per_block;
    const int block_offset = linear_idx - blk_local * words_per_block;
    const int kk_word = block_offset / kStep1RowTile;
    const int rr = block_offset - kk_word * kStep1RowTile;
    uint32_t value = 0u;
    if (rr < n_rows) {
      const int packed_idx = packed_base + row_tile + rr;
      const int token_idx = valid_token_idx[packed_idx];
      const int k_blk = k_blk_base + blk_local;
      const int k0_word = (k_blk * kStep1Block) / 4;
      const int word_idx = static_cast<int64_t>(token_idx) * kHiddenWords + k0_word + kk_word;
      value = hidden_words_dev[word_idx];
    }
    smem_B_words[rr * (k_blk_count * kWordsPerKBlock) + blk_local * kWordsPerKBlock + kk_word] = value;
  }
}

__device__ __forceinline__ bool issue_weight_group_tma_combined(
    uint32_t* smem_weight_group,
    const void* __restrict__ w13_tma_desc,
    int expert,
    int out_row_base,
    int out_rows,
    int h_tiles,
    int k_group,
    uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (w13_tma_desc == nullptr) return false;
  const uint32_t kBytesPerChunk =
      2 * out_rows * kStep1Block * h_tiles;
  if (threadIdx.x == 0) {
    const int32_t coords[5] = {0, k_group * h_tiles, out_row_base, 0, expert};
    cuda::ptx::mbarrier_init(barrier, 1);
    cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global, cuda::ptx::cta_group_1,
        smem_weight_group, w13_tma_desc, coords, barrier);
    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared,
        barrier, kBytesPerChunk);
  }
  return true;
#else
  (void)smem_weight_group;
  (void)w13_tma_desc;
  (void)expert;
  (void)out_row_base;
  (void)out_rows;
  (void)h_tiles;
  (void)k_group;
  (void)barrier;
  return false;
#endif
}

__device__ __forceinline__ void wait_weight_group_tma_combined(uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (threadIdx.x == 0) {
    while (!cuda::ptx::mbarrier_try_wait_parity(
        cuda::ptx::sem_acquire, cuda::ptx::scope_cta, barrier, 0u)) {}
  }
  __syncthreads();
#else
  (void)barrier;
#endif
}

// -----------------------------------------------------------------------------
// Lane-local accumulator fragment for the restructured fallback path.
// Each lane owns exactly one output column inside the 32-row CTA tile and all
// 8 token rows of the current row tile.
// -----------------------------------------------------------------------------
struct LaneAccum8x1 {
  float v[kStep1RowTile];

  __device__ __forceinline__ void clear() {
#pragma unroll
    for (int r = 0; r < kStep1RowTile; ++r) {
      v[r] = 0.0f;
    }
  }
};

__device__ __forceinline__ void mma_64x8x64_tt_fallback(
    LaneAccum8x1& gate_acc,
    LaneAccum8x1& up_acc,
    const uint32_t* __restrict__ smem_A_group_combined_words,
    const uint32_t* __restrict__ smem_B_group_words,
    float gate_block_scale,
    float up_block_scale,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int row_tile,
    int n_rows,
    int k_blk,
    int k_blk_local,
    int k64_sub,
    int debug_output_mode,
    int lane);

template <class TensorA, class TensorB>
__device__ __forceinline__ void stage_tcgen_operands(
    TensorA sA_tcgen,
    TensorB sB_tcgen,
    const uint8_t* __restrict__ smem_A_tile_bytes,
    const uint8_t* __restrict__ smem_B_group_bytes,
    int a_row_stride_bytes,
    int a_byte_base,
    int b_byte_base,
    int lane) {
  for (int linear_idx = lane; linear_idx < kStep1TcgenACompactBytes; linear_idx += kStep1Threads) {
    const int m = linear_idx / kStep1TcgenSubKBytes;
    const int k = linear_idx - m * kStep1TcgenSubKBytes;
    sA_tcgen(cute::make_coord(m, k), 0, 0) =
        smem_A_tile_bytes[m * a_row_stride_bytes + a_byte_base + k];
  }

  for (int linear_idx = lane; linear_idx < kStep1TcgenBCompactBytes; linear_idx += kStep1Threads) {
    const int n = linear_idx / kStep1TcgenSubKBytes;
    const int k = linear_idx - n * kStep1TcgenSubKBytes;
    sB_tcgen(cute::make_coord(n, k), 0, 0) =
        smem_B_group_bytes[n * kStep1HiddenGroupBytes + b_byte_base + k];
  }
}

template <class TensorA, class TensorB>
__device__ __forceinline__ void stage_tcgen_operands(
    TensorA sA_tcgen,
    TensorB sB_tcgen,
    const uint32_t* __restrict__ smem_A_group_combined_words,
    const uint32_t* __restrict__ smem_B_group_words,
    int k_blk_local,
    int k64_sub,
    int lane) {
  const int a_byte_base = k_blk_local * kStep1Block + k64_sub * kStep1TcgenSubKBytes;
  const int b_byte_base = a_byte_base;
  stage_tcgen_operands(
      sA_tcgen,
      sB_tcgen,
      reinterpret_cast<const uint8_t*>(smem_A_group_combined_words),
      reinterpret_cast<const uint8_t*>(smem_B_group_words),
      kStep1HiddenGroupBytes,
      a_byte_base,
      b_byte_base,
      lane);
}

#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) && \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ uint64_t pack_umma_desc_u64(cute::UMMA::SmemDescriptor const& desc) {
  return (static_cast<uint64_t>(desc.hi) << 32) | static_cast<uint64_t>(desc.lo);
}

__device__ __forceinline__ constexpr uint32_t make_tcgen05_idesc_f8f6f4_f32_dense(int m, int n) {
  return (1u << 4) | (static_cast<uint32_t>(n >> 3) << 17) | (static_cast<uint32_t>(m >> 4) << 24);
}

__device__ __forceinline__ uint64_t make_tcgen05_smem_desc_major_k_u8(
    const void* smem_ptr, int row_stride_bytes) {
  const uint32_t smem = cute::cast_smem_ptr_to_uint(smem_ptr);
  const uint32_t matrix_start_aligned = smem & ~0xFu;
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>(matrix_start_aligned >> 4);
  desc |= static_cast<uint64_t>((16u & 0x3ffffu) >> 4) << 16;
  desc |= static_cast<uint64_t>(((static_cast<uint32_t>(row_stride_bytes) * 8u) & 0x3ffffu) >> 4) << 32;
  desc |= static_cast<uint64_t>(1u) << 46;
  desc |= static_cast<uint64_t>(0xb0u) << 53;
  return desc;
}

__device__ __forceinline__ void tcgen05_mma_f8f6f4_cta1_ss(
    uint32_t d_tmem,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t idesc,
    bool enable_input_d) {
  uint32_t enable_u32 = enable_input_d ? 1u : 0u;
  uint32_t disable_output_lane[4] = {0, 0, 0, 0};
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.u32 p, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%4, %5, %6, %7}, p;\n\t"
      "}\n"
      :
      : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
        "r"(disable_output_lane[0]), "r"(disable_output_lane[1]),
        "r"(disable_output_lane[2]), "r"(disable_output_lane[3]), "r"(enable_u32)
      : "memory");
}

__device__ __forceinline__ void tcgen05_commit_group1(uint64_t const* smem_ptr) {
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :
               : "r"(bar_intptr)
               : "memory");
}

__device__ __forceinline__ void tcgen05_wait_ld_sync() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" : : : "memory");
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x8(float (&dst)[8], uint32_t taddr) {
  uint32_t bits[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
               "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(bits[0]), "=r"(bits[1]), "=r"(bits[2]), "=r"(bits[3]),
                 "=r"(bits[4]), "=r"(bits[5]), "=r"(bits[6]), "=r"(bits[7])
               : "r"(taddr)
               : "memory");
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    dst[i] = __uint_as_float(bits[i]);
  }
}

__device__ __forceinline__ void tcgen05_ld_16x64b_x4(float (&dst)[4], uint32_t taddr) {
  uint32_t bits[4];
  asm volatile("tcgen05.ld.sync.aligned.16x64b.x4.b32 "
               "{%0, %1, %2, %3}, [%4];"
               : "=r"(bits[0]), "=r"(bits[1]), "=r"(bits[2]), "=r"(bits[3])
               : "r"(taddr)
               : "memory");
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    dst[i] = __uint_as_float(bits[i]);
  }
}

__device__ __forceinline__ bool tcgen05_wait_mma_barrier(uint64_t* barrier, int phase, int lane) {
  const uint32_t bar_ptr = cute::cast_smem_ptr_to_uint(barrier);
  uint32_t ready = 0;
  if (lane == 0) {
#pragma unroll 1
    while (!ready) {
      asm volatile(
          "{ .reg .pred p;"
          "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
          "  selp.u32 %0, 1, 0, p; }"
          : "=r"(ready)
          : "r"(bar_ptr), "r"(static_cast<uint32_t>(phase))
          : "memory");
    }
  }
  ready = __shfl_sync(0xffffffffu, ready, 0);
  return ready != 0;
}

__device__ __forceinline__ void tcgen05_wait_mma_barrier_single(uint64_t* barrier, int phase) {
  const uint32_t bar_ptr = cute::cast_smem_ptr_to_uint(barrier);
  uint32_t ready = 0;
  while (!ready) {
    asm volatile(
        "{ .reg .pred p;"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
        "  selp.u32 %0, 1, 0, p; }"
        : "=r"(ready)
        : "r"(bar_ptr), "r"(static_cast<uint32_t>(phase))
        : "memory");
  }
}

__device__ __forceinline__ void tcgen05_mbarrier_init(uint64_t* barrier, uint32_t count) {
  const uint32_t bar_ptr = cute::cast_smem_ptr_to_uint(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :: "r"(bar_ptr), "r"(count)
               : "memory");
}

__device__ __forceinline__ uint32_t tcgen05_alloc_cta1_cols_32(uint32_t* smem_out_taddr) {
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_out_taddr);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
               :: "r"(smem_addr)
               : "memory");
  __syncwarp();
  uint32_t taddr;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr) : "r"(smem_addr) : "memory");
  return taddr;
}

__device__ __forceinline__ void tcgen05_dealloc_cta1_cols_32(uint32_t taddr) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
               :: "r"(taddr)
               : "memory");
}

__device__ __forceinline__ void tcgen05_relinquish_alloc_permit_cta1() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

template <class TensorA, class TensorB>
__device__ __forceinline__ void stage_tcgen05_compact_ab_64x8x64(
    TensorA smem_A_compact,
    TensorB smem_B_compact,
    const uint32_t* __restrict__ smem_A_group_combined_words,
    const uint32_t* __restrict__ smem_B_group_words,
    int k_blk_local,
    int k64_sub,
    int mma_sub,
    int lane) {
  const uint8_t* smem_A_bytes = reinterpret_cast<const uint8_t*>(smem_A_group_combined_words);
  const uint8_t* smem_B_bytes = reinterpret_cast<const uint8_t*>(smem_B_group_words);
  const int byte_base =
      k_blk_local * kStep1Block + k64_sub * kStep1TcgenSubKBytes + mma_sub * kStep1TcgenMmaKBytes;
  const int up_group_base = kStep1OutRowsPerCta * kStep1HiddenGroupBytes;
  constexpr int kAStageBytes = kStep1TcgenM * kStep1TcgenCompactKBytes;
  constexpr int kBStageBytes = kStep1TcgenN * kStep1TcgenCompactKBytes;

  for (int linear = threadIdx.x; linear < kAStageBytes; linear += kStep1CommThreads) {
    const int m = linear / kStep1TcgenCompactKBytes;
    const int k = linear - m * kStep1TcgenCompactKBytes;
    const int slice = k >> 4;
    const int byte = k & 15;
    const int src = (m < kStep1OutRowsPerCta)
                        ? (m * kStep1HiddenGroupBytes + byte_base + k)
                        : (up_group_base + (m - kStep1OutRowsPerCta) * kStep1HiddenGroupBytes +
                           byte_base + k);
    smem_A_compact[slice * kStep1TcgenM * 16 + m * 16 + byte] = smem_A_bytes[src];
  }

  for (int linear = threadIdx.x; linear < kBStageBytes; linear += kStep1CommThreads) {
    const int n = linear / kStep1TcgenCompactKBytes;
    const int k = linear - n * kStep1TcgenCompactKBytes;
    const int slice = k >> 4;
    const int byte = k & 15;
    smem_B_compact[slice * kStep1TcgenN * 16 + n * 16 + byte] =
        smem_B_bytes[n * kStep1HiddenGroupBytes + byte_base + k];
  }
}

__device__ __forceinline__ uint64_t make_tcgen05_core_matrix_desc(const void* smem, int height) {
  const uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem);
  const uint32_t matrix_start_aligned = smem_addr & ~0xFu;
  const uint32_t lbo = static_cast<uint32_t>(height * 16);
  const uint32_t sbo = 8u * 16u;
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>(matrix_start_aligned >> 4);
  desc |= static_cast<uint64_t>((lbo & 0x3ffffu) >> 4) << 16;
  desc |= static_cast<uint64_t>((sbo & 0x3ffffu) >> 4) << 32;
  desc |= static_cast<uint64_t>(1u) << 46;
  desc |= static_cast<uint64_t>(0xb0u) << 53;
  return desc;
}
#endif

__device__ __forceinline__ void mma_64x8x64_tcgen05(
    LaneAccum8x1& gate_acc,
    LaneAccum8x1& up_acc,
    uint8_t* smem_A_tcgen_bytes,
    uint8_t* smem_B_tcgen_bytes,
    const uint32_t* __restrict__ smem_A_group_combined_words,
    const uint32_t* __restrict__ smem_B_group_words,
    float gate_block_scale,
    float up_block_scale,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int row_tile,
    int n_rows,
    int k_blk,
    int k_blk_local,
    int k64_sub,
    int lane,
    int warp_id,
    int debug_output_mode,
    uint32_t tmem_base_ptr,
    uint64_t* mma_barrier,
    int& mma_phase_bit) {
#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) && \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t result_tmem = tmem_base_ptr;
  const uint32_t idesc = make_tcgen05_idesc_f8f6f4_f32_dense(kStep1TcgenM, kStep1TcgenN);
  const uint64_t a_desc = make_tcgen05_core_matrix_desc(smem_A_tcgen_bytes, kStep1TcgenM);
  const uint64_t b_desc = make_tcgen05_core_matrix_desc(smem_B_tcgen_bytes, kStep1TcgenN);

#pragma unroll
  for (int mma_sub = 0; mma_sub < 2; ++mma_sub) {
    stage_tcgen05_compact_ab_64x8x64(
        smem_A_tcgen_bytes, smem_B_tcgen_bytes, smem_A_group_combined_words, smem_B_group_words,
        k_blk_local, k64_sub, mma_sub, lane);
    __syncthreads();
    if (warp_id == 0 && lane == 0) {
      tcgen05_mma_f8f6f4_cta1_ss(result_tmem, a_desc, b_desc, idesc,
                                  (k64_sub != 0) || (mma_sub != 0));
      tcgen05_commit_group1(mma_barrier);
    }
    if (threadIdx.x == 0) {
      tcgen05_wait_mma_barrier_single(mma_barrier, mma_phase_bit);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    }
    __syncthreads();
    mma_phase_bit ^= 1;
    if (k64_sub != 1 || mma_sub != 1) {
      continue;
    }
      if (warp_id < 4) {
      float result_bits[4];
      tcgen05_ld_16x64b_x4(result_bits, result_tmem);
      tcgen05_wait_ld_sync();

      const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
      const int col_parity = (lane >> 1) & 1;
	#pragma unroll
        for (int reg = 0; reg < 4; ++reg) {
          const int rr = 2 * reg + col_parity;
          if (rr >= n_rows) continue;
          const int packed_idx = packed_base + row_tile + rr;
          const int token_idx = valid_token_idx[packed_idx];
          const float hidden_block_scale =
              hidden_scale_dev[static_cast<int64_t>(k_blk) * t + token_idx];
          const bool dump_unscaled_gate = debug_output_mode == 3;
          const bool dump_unscaled_up = debug_output_mode == 4;
          if (result_row < kStep1OutRowsPerCta) {
            const float scale = dump_unscaled_gate ? 1.0f : gate_block_scale * hidden_block_scale;
            gate_acc.v[rr] += result_bits[reg] * scale;
          } else {
            const float scale = dump_unscaled_up ? 1.0f : up_block_scale * hidden_block_scale;
            up_acc.v[rr] += result_bits[reg] * scale;
          }
        }
    }
  }
#else
  (void)gate_acc;
  (void)up_acc;
  (void)smem_A_tcgen_bytes;
  (void)smem_B_tcgen_bytes;
  (void)smem_A_group_combined_words;
  (void)smem_B_group_words;
  (void)gate_block_scale;
  (void)up_block_scale;
  (void)hidden_scale_dev;
  (void)t;
  (void)valid_token_idx;
  (void)packed_base;
  (void)row_tile;
  (void)n_rows;
  (void)k_blk;
  (void)k_blk_local;
  (void)k64_sub;
  (void)lane;
  (void)warp_id;
  (void)debug_output_mode;
  (void)tmem_base_ptr;
  (void)mma_barrier;
  (void)mma_phase_bit;
#endif
}

// -----------------------------------------------------------------------------
// Fallback compute for a 64x8x64 grouped microkernel.
// Shared-memory physical layout:
//   A_group_combined: [2][32][1024] with dim order [gate_up][row][k_byte]
//   B_group:          [8][1024] row-major with K contiguous in memory
// Each call consumes one 64-byte K-subgroup and updates both gate/up fragments.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void mma_64x8x64_tt_fallback(
    LaneAccum8x1& gate_acc,
    LaneAccum8x1& up_acc,
    const uint32_t* __restrict__ smem_A_group_combined_words,
    const uint32_t* __restrict__ smem_B_group_words,
    float gate_block_scale,
    float up_block_scale,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int row_tile,
    int n_rows,
    int k_blk,
    int k_blk_local,
    int k64_sub,
    int debug_output_mode,
    int lane) {
  constexpr int kAGroupWordsPerRow = kStep1HiddenGroupWords;   // 256
  constexpr int kAHalfWords = kStep1OutRowsPerCta * kAGroupWordsPerRow;
  constexpr int kWordsPer64 = 64 / 4;
  const int a_word_base = k_blk_local * (kStep1Block / 4) + k64_sub * kWordsPer64;
  const int b_word_base = a_word_base;
  const int gate_row_base = lane * kAGroupWordsPerRow + a_word_base;
  const int up_row_base = kAHalfWords + lane * kAGroupWordsPerRow + a_word_base;

#pragma unroll
  for (int rr = 0; rr < kStep1RowTile; ++rr) {
    if (rr >= n_rows) break;

    const int packed_idx = packed_base + row_tile + rr;
    const int token_idx = valid_token_idx[packed_idx];
    const float hidden_block_scale =
        hidden_scale_dev[static_cast<int64_t>(k_blk) * t + token_idx];
    const bool dump_unscaled_gate = debug_output_mode == 3;
    const bool dump_unscaled_up = debug_output_mode == 4;
    const float gate_fused_scale = dump_unscaled_gate ? 1.0f : gate_block_scale * hidden_block_scale;
    const float up_fused_scale = dump_unscaled_up ? 1.0f : up_block_scale * hidden_block_scale;

    float gate_partial = 0.0f;
    float up_partial = 0.0f;
#pragma unroll
    for (int kk_word = 0; kk_word < kWordsPer64; ++kk_word) {
      const uint32_t gate_word = smem_A_group_combined_words[gate_row_base + kk_word];
      const uint32_t up_word = smem_A_group_combined_words[up_row_base + kk_word];
      const uint32_t b_word =
          smem_B_group_words[rr * kStep1HiddenGroupWords + (b_word_base + kk_word)];
#pragma unroll
      for (int byte_idx = 0; byte_idx < 4; ++byte_idx) {
        const float a_gate = fp8_e4m3fn_to_float_device(extract_byte(gate_word, byte_idx));
        const float a_up = fp8_e4m3fn_to_float_device(extract_byte(up_word, byte_idx));
        const float b = fp8_e4m3fn_to_float_device(extract_byte(b_word, byte_idx));
        gate_partial += a_gate * b;
        up_partial += a_up * b;
      }
    }
    gate_acc.v[rr] += gate_partial * gate_fused_scale;
    up_acc.v[rr] += up_partial * up_fused_scale;
  }
}

// -----------------------------------------------------------------------------
// PTX hook point.
// The CTA structure now matches the communication-only kernel:
//   one CTA = one expert x one 32-row output tile
//   one K-group = 8 contiguous 128-byte hidden blocks = 1024-byte slice
// Replace mma_32x8x32_tt_fallback() with tcgen05 once the fragment mapping,
// tmem allocation, and wait semantics are finalized.
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
    int debug_output_mode,
    float* __restrict__ c_perm_all_dev) {
  (void)hidden_tma_desc;

  const int out_tile32 = blockIdx.x;  // 0..63
  const int expert = blockIdx.y;      // 0..31
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  if (expert >= kStep1LocalExperts) return;
  if (out_tile32 >= kStep1OutTilesPerExpert) return;

  const int out_row_base = out_tile32 * kStep1OutRowsPerCta;
  const int out_block128 = out_row_base / kStep1Block;

  const int t_valid = expert_t_valid[expert];
  if (t_valid <= 0) return;

  const int packed_base = expert_offset[expert];
  const uint4* w13_all_vec_dev = reinterpret_cast<const uint4*>(w13_all_dev);
  const uint32_t* hidden_words_dev = reinterpret_cast<const uint32_t*>(hidden_fp8_dev);

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const std::size_t weight_group_bytes =
      static_cast<std::size_t>(2) * kStep1OutRowsPerCta * kStep1HiddenGroupBytes;
  const std::size_t hidden_group_offset = weight_group_bytes;
  const std::size_t tcgen_a_offset =
      align_up(hidden_group_offset + static_cast<std::size_t>(kStep1HiddenGroupBytes) * kStep1RowTile, 128);
  const std::size_t tcgen_b_offset = align_up(tcgen_a_offset + kStep1TcgenACompactBytes, 128);
  uint32_t* smem_A_group = reinterpret_cast<uint32_t*>(smem_raw);
  uint32_t* smem_B_group = reinterpret_cast<uint32_t*>(smem_raw + hidden_group_offset);
  uint8_t* smem_A_tcgen = smem_raw + tcgen_a_offset;
  uint8_t* smem_B_tcgen = smem_raw + tcgen_b_offset;
  alignas(16) __shared__ float tcgen_gate_acc_smem[kStep1OutRowsPerCta * kStep1RowTile];
  alignas(16) __shared__ float tcgen_up_acc_smem[kStep1OutRowsPerCta * kStep1RowTile];

#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  alignas(16) __shared__ uint64_t weight_group_barrier;
  alignas(16) __shared__ uint64_t tcgen_mma_barrier;
  alignas(16) __shared__ uint32_t tcgen_tmem_base_ptr;
  if (threadIdx.x == 0) {
    tcgen05_mbarrier_init(&tcgen_mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_cta1_cols_32(&tcgen_tmem_base_ptr);
    if (lane == 0) {
      tcgen_tmem_base_ptr = taddr;
    }
  }
  __syncthreads();
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  alignas(16) __shared__ uint64_t weight_group_barrier;
#endif

  for (int row_tile = 0; row_tile < t_valid; row_tile += kStep1RowTile) {
    const int n_rows = min_int(t_valid - row_tile, kStep1RowTile);

    for (int idx = threadIdx.x; idx < kStep1OutRowsPerCta * kStep1RowTile; idx += blockDim.x) {
      tcgen_gate_acc_smem[idx] = 0.0f;
      tcgen_up_acc_smem[idx] = 0.0f;
    }
    __syncthreads();

    LaneAccum8x1 gate_acc;
    LaneAccum8x1 up_acc;
    if (warp_id < 4) {
      gate_acc.clear();
      up_acc.clear();
    }

    const bool use_tma = (w13_tma_desc != nullptr);
    int tcgen_mma_phase_bit = 0;

    for (int k_group = 0; k_group < kStep1TmaRawChunkGroups; ++k_group) {
      if (use_tma) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        issue_weight_group_tma_combined(
            smem_A_group, w13_tma_desc, expert, out_row_base, kStep1OutRowsPerCta,
            kStep1TmaRawChunkBlocks, k_group, &weight_group_barrier);
#else
        load_weight_group_vec_combined<kStep1CommThreads>(
            smem_A_group, w13_all_vec_dev, expert, out_row_base, k_group);
#endif
      } else {
        load_weight_group_vec_combined<kStep1CommThreads>(
            smem_A_group, w13_all_vec_dev, expert, out_row_base, k_group);
      }

      load_hidden_wordtiles_group<kStep1CommThreads>(
          smem_B_group, hidden_words_dev, valid_token_idx, packed_base, row_tile, n_rows,
          k_group * kStep1TmaRawChunkBlocks, kStep1TmaRawChunkBlocks);

      __syncthreads();

      if (use_tma) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        wait_weight_group_tma_combined(&weight_group_barrier);
#endif
      }

#pragma unroll
      for (int k_blk_local = 0; k_blk_local < kStep1TmaRawChunkBlocks; ++k_blk_local) {
        const int k_blk = k_group * kStep1TmaRawChunkBlocks + k_blk_local;
        const float gate_scale =
            s13_all_dev[(expert * kStep1Gemm1OutBlocks + out_block128) * kStep1HiddenBlocks + k_blk];
        const float up_scale =
            s13_all_dev[(expert * kStep1Gemm1OutBlocks +
                         (out_block128 + kStep1IntermediateBlocks)) * kStep1HiddenBlocks + k_blk];
#pragma unroll
        for (int k64_sub = 0; k64_sub < 2; ++k64_sub) {
          mma_64x8x64_tcgen05(
              gate_acc,
              up_acc,
              smem_A_tcgen,
              smem_B_tcgen,
              smem_A_group,
              smem_B_group,
              gate_scale,
              up_scale,
              hidden_scale_dev,
              t,
              valid_token_idx,
              packed_base,
              row_tile,
              n_rows,
              k_blk,
              k_blk_local,
              k64_sub,
              lane,
              warp_id,
              debug_output_mode,
#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              tcgen_tmem_base_ptr,
              &tcgen_mma_barrier,
              tcgen_mma_phase_bit
#else
              0u,
              nullptr,
              tcgen_mma_phase_bit
#endif
              );
        }
      }

      __syncthreads();
    }

    if (warp_id < 4) {
      const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
      const int col_parity = (lane >> 1) & 1;
#pragma unroll
      for (int reg = 0; reg < 4; ++reg) {
        const int rr = 2 * reg + col_parity;
        if (rr >= n_rows) continue;
        if (result_row < kStep1OutRowsPerCta) {
          tcgen_gate_acc_smem[result_row * kStep1RowTile + rr] = gate_acc.v[rr];
        } else {
          tcgen_up_acc_smem[(result_row - kStep1OutRowsPerCta) * kStep1RowTile + rr] = up_acc.v[rr];
        }
      }
    }
    __syncthreads();

    const int out_col = out_row_base + lane;
    const bool do_store = (warp_id == 0);

    if (do_store) {
#pragma unroll
      for (int rr = 0; rr < kStep1RowTile; ++rr) {
        if (rr >= n_rows) break;
        const int packed_row = packed_base + row_tile + rr;
        const int tcgen_row = lane;
        const float gate_v = tcgen_gate_acc_smem[tcgen_row * kStep1RowTile + rr];
        const float up_v = tcgen_up_acc_smem[tcgen_row * kStep1RowTile + rr];
        const float value = (debug_output_mode == 1 || debug_output_mode == 3) ? gate_v :
                            (debug_output_mode == 2 || debug_output_mode == 4) ? up_v :
                            gate_v * silu_device(up_v);
        c_perm_all_dev[static_cast<int64_t>(packed_row) * kStep1Intermediate + out_col] = value;
      }
    }
    __syncthreads();
  }

#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (warp_id == 0) {
    tcgen05_dealloc_cta1_cols_32(tcgen_tmem_base_ptr);
  }
  __syncthreads();
  if (warp_id == 0) {
    tcgen05_relinquish_alloc_permit_cta1();
  }
  __syncthreads();
#endif
}

inline cudaError_t RunStep1AllExpertsDirect(
    const uint8_t* hidden_fp8_dev,
    const float* hidden_scale_dev,
    int64_t t,
    const int* expert_counts_dev,
    const int* expert_offsets_dev,
    const int* permuted_token_ids_dev,
    const uint8_t* w13_all_dev,
    const float* s13_all_dev,
    const void* w13_tma_desc,
    int debug_output_mode,
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

  dim3 grid(kStep1OutTilesPerExpert, kStep1LocalExperts, 1);
  dim3 block(kStep1CommThreads, 1, 1);
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
      w13_tma_desc,
      debug_output_mode,
      c_perm_all_dev);

  return cudaGetLastError();
}

template <bool kUseTma>
static __global__ void step1_comm_only_direct_kernel(
    const uint8_t* __restrict__ hidden_bytes_dev,
    int64_t t,
    const int* __restrict__ expert_t_valid,
	    const int* __restrict__ expert_offset,
	    const int* __restrict__ valid_token_idx,
	    const uint8_t* __restrict__ w13_all_dev,
	    const void* __restrict__ w13_tma_desc,
	    int comm_tma_mode,
	        int comm_h_inner_bytes,
	        int comm_out_rows,
	        int comm_h_tiles,
	        bool comm_double_buffer,
	        bool comm_skip_hidden,
	        bool comm_skip_weight) {
  (void)comm_tma_mode;
  (void)comm_double_buffer;
  const int n_tile = blockIdx.x;
  const int expert = blockIdx.y;
  const int lane = threadIdx.x & 31;
  if (expert >= kStep1LocalExperts) return;
  const int n_tile_base_row = n_tile * comm_out_rows;
  if (n_tile_base_row >= kStep1Intermediate) return;

  const int t_valid = expert_t_valid[expert];
  if (t_valid <= 0) return;

  const int packed_base = expert_offset[expert];

	  extern __shared__ __align__(1024) uint8_t smem_raw[];
	  const std::size_t weight_group_bytes =
	      static_cast<std::size_t>(2) * comm_out_rows * comm_h_inner_bytes * comm_h_tiles;
	  uint32_t* smem_A_group = reinterpret_cast<uint32_t*>(smem_raw);
	  uint32_t* smem_B_words = reinterpret_cast<uint32_t*>(smem_raw + weight_group_bytes);
	  const uint4* w13_all_vec_dev = reinterpret_cast<const uint4*>(w13_all_dev);
	  const uint32_t* hidden_words_dev = reinterpret_cast<const uint32_t*>(hidden_bytes_dev);
	  alignas(16) __shared__ uint64_t weight_group_barrier;
	  volatile uint32_t sink = 0;
	  const bool used_tma = kUseTma && (w13_tma_desc != nullptr) && !comm_skip_weight;
	  const int comm_tma_groups = kStep1HiddenBlocks / comm_h_tiles;

	  for (int row_tile = 0; row_tile < t_valid; row_tile += kStep1RowTile) {
	    const int n_rows = min_int(t_valid - row_tile, kStep1RowTile);
	    for (int k_group = 0; k_group < comm_tma_groups; ++k_group) {
	      if (!comm_skip_weight) {
	        if (used_tma) {
	          issue_weight_group_tma_combined(
	              smem_A_group, w13_tma_desc, expert, n_tile_base_row, comm_out_rows,
	              comm_h_tiles, k_group, &weight_group_barrier);
	        } else {
	          load_weight_wordtiles_group_combined<kStep1CommThreads>(
	              smem_A_group, w13_all_vec_dev, expert, n_tile_base_row, comm_out_rows,
	              k_group * comm_h_tiles, comm_h_tiles);
	        }
	      }
	      if (!comm_skip_hidden) {
	        load_hidden_wordtiles_group<kStep1CommThreads>(
	            smem_B_words, hidden_words_dev, valid_token_idx, packed_base, row_tile, n_rows,
	            k_group * comm_h_tiles, comm_h_tiles);
	      }
	      __syncthreads();
	      if (used_tma) {
	        wait_weight_group_tma_combined(&weight_group_barrier);
	      }
	      if (threadIdx.x < 32) {
	        const int weight_words = static_cast<int>(weight_group_bytes / 4);
	        const int hidden_words = comm_h_tiles * (kStep1Block / 4) * kStep1RowTile;
	        const uint32_t a = comm_skip_weight ? 0u : smem_A_group[(threadIdx.x * 17) % weight_words];
	        const uint32_t b = comm_skip_hidden ? 0u : smem_B_words[(threadIdx.x * 13) % hidden_words];
	        sink ^= (a + b + static_cast<uint32_t>(lane));
	      }
	      __syncthreads();
	    }
	  }
	  if (threadIdx.x == 0 && sink == 0xffffffffu) {
	    printf("step1_comm_only_sink\n");
	  }
}

inline cudaError_t RunStep1CommOnlyDirect(
    const uint8_t* hidden_fp8_dev,
    int64_t t,
    const int* expert_counts_dev,
	    const int* expert_offsets_dev,
	    const int* permuted_token_ids_dev,
			    const uint8_t* w13_all_dev,
			    const void* w13_tma_desc,
                int comm_tma_mode,
                int comm_h_inner_bytes,
                int comm_out_rows,
                int comm_h_tiles,
                bool comm_double_buffer,
                bool comm_skip_hidden,
                bool comm_skip_weight,
				    cudaStream_t stream) {
		  const std::size_t smem_bytes =
	          step1_tma_comm_smem_bytes(comm_h_inner_bytes, comm_out_rows, comm_h_tiles, false);
		  const std::size_t tma_smem_bytes =
	          step1_tma_comm_smem_bytes(comm_h_inner_bytes, comm_out_rows, comm_h_tiles,
	                                    comm_double_buffer);

	  cudaError_t st = cudaFuncSetAttribute(
	      step1_comm_only_direct_kernel<false>,
	      cudaFuncAttributeMaxDynamicSharedMemorySize,
	      static_cast<int>(smem_bytes));
  if (st != cudaSuccess) {
    return st;
  }
	  st = cudaFuncSetAttribute(
	      step1_comm_only_direct_kernel<true>,
	      cudaFuncAttributeMaxDynamicSharedMemorySize,
	      static_cast<int>(tma_smem_bytes));
  if (st != cudaSuccess) {
    return st;
  }

	  dim3 grid(kStep1Intermediate / comm_out_rows, kStep1LocalExperts, 1);
	  dim3 block_vec(kStep1CommThreads, 1, 1);
	  dim3 block_tma(kStep1CommThreads, 1, 1);

			  if (w13_tma_desc != nullptr) {
			    step1_comm_only_direct_kernel<true><<<grid, block_tma, tma_smem_bytes, stream>>>(
			        hidden_fp8_dev, t, expert_counts_dev, expert_offsets_dev, permuted_token_ids_dev,
			        w13_all_dev, w13_tma_desc, comm_tma_mode, comm_h_inner_bytes,
                    comm_out_rows, comm_h_tiles,
                    comm_double_buffer, comm_skip_hidden, comm_skip_weight);
		  } else {
		    step1_comm_only_direct_kernel<false><<<grid, block_vec, smem_bytes, stream>>>(
		        hidden_fp8_dev, t, expert_counts_dev, expert_offsets_dev, permuted_token_ids_dev,
		        w13_all_dev, nullptr, 0, comm_h_inner_bytes, comm_out_rows, comm_h_tiles,
                false, comm_skip_hidden, comm_skip_weight);
		  }

  return cudaGetLastError();
}

}  // namespace direct_backend
