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
static constexpr int kStep1CommThreads = 128;                                     // four warps for cooperative staging
static constexpr int kStep1CommWarps = kStep1CommThreads / kStep1Threads;
static constexpr int kStep1HiddenBlocks = kStep1Hidden / kStep1Block;             // 56
static constexpr int kStep1TmaChunkBlocks = 4;
static constexpr int kStep1TmaChunkGroups = kStep1HiddenBlocks / kStep1TmaChunkBlocks;
static constexpr int kStep1TmaRawChunkBlocks = 4;                                 // 512-byte K-group
static constexpr int kStep1TmaRawChunkGroups = kStep1HiddenBlocks / kStep1TmaRawChunkBlocks;  // 14
static constexpr int kStep1IntermediateBlocks = kStep1Intermediate / kStep1Block; // 16
static constexpr int kStep1Gemm1OutBlocks = kStep1Gemm1Out / kStep1Block;         // 32
static constexpr int kStep1KSubsPerBlock = kStep1Block / kStep1TcgenK;            // 4
static constexpr int kStep1OutTilesPerExpert = kStep1Intermediate / kStep1OutRowsPerCta;       // 64
static constexpr int kStep1RowsPer128Block = kStep1Block / kStep1OutRowsPerCta;                // 4
static constexpr int kStep1HiddenGroupBytes = kStep1Block * kStep1TmaRawChunkBlocks;           // 512
static constexpr int kStep1HiddenGroupWords = kStep1HiddenGroupBytes / 4;                       // 256
static constexpr int kStep1TcgenMmaKBytes = 32;
static constexpr int kStep1TcgenSubKBytes = 64;
static constexpr int kStep1TcgenM = 64;
static constexpr int kStep1TcgenN = 8;
static constexpr int kStep1TcgenTmemCols = 64;
static constexpr int kStep1TcgenCompactKBytes = 32;
static constexpr int kStep1TcgenACompactBytes = kStep1TcgenM * kStep1Block;                    // 8192
static constexpr int kStep1TcgenBCompactBytes = kStep1TcgenN * kStep1Block;                    // 1024
static constexpr int kStep1TcgenScaleDiagBytes =
    2 * kStep1HiddenBlocks * 2 * kStep1TcgenN * 4 * sizeof(float);                              // 28672
static constexpr uint32_t kStep1TcgenLayoutSwizzle128B = 2;
static constexpr int kStep1PipelineStages = 2;
static constexpr int kStep1ExchangeSmemElems = kStep1RowTile * kStep1OutRowsPerCta;

static_assert(kStep1Hidden % kStep1Block == 0, "Hidden size must be divisible by 128.");
static_assert(kStep1Block % kStep1TcgenK == 0, "128-block must be divisible by tcgen05 K=32.");
static_assert(kStep1CommThreads % kStep1Threads == 0, "Comm threads must be whole warps.");
static_assert(kStep1HiddenBlocks % kStep1TmaChunkBlocks == 0, "TMA chunking requires exact division.");
static_assert(kStep1HiddenBlocks % kStep1TmaRawChunkBlocks == 0, "K grouping requires exact division.");
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

__device__ __forceinline__ __half2 silu_half2_device(__half2 x) {
  const __half2 one = __float2half2_rn(1.0f);
  return __hmul2(x, h2rcp(__hadd2(one, h2exp(__hneg2(x)))));
}

__device__ __forceinline__ __half2 swiglu_half2_device(__half2 gate, __half2 up) {
  return __hmul2(gate, silu_half2_device(up));
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
  offset += kStep1PipelineStages * weight_group_bytes;
  offset += kStep1PipelineStages * hidden_group_bytes;
  offset = align_up(offset, 1024);
  offset += kStep1TcgenACompactBytes;
  offset = align_up(offset, 1024);
  offset += kStep1TcgenBCompactBytes;
  offset = align_up(offset, 1024);
  offset += kStep1TcgenScaleDiagBytes;
  return align_up(offset + 1024, 1024);
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
  return align_up(offset + 1024, 1024);
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

#pragma unroll
  for (int iter = 0; iter < (kStep1OutRowsPerCta * kGroupVecsPerRow + kThreads - 1) / kThreads;
       ++iter) {
    const int chunk_idx = threadIdx.x + iter * kThreads;
    if (chunk_idx >= total_chunks) continue;
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
  constexpr int kMaxTotalChunks = kStep1Block * kStep1TmaRawChunkBlocks * kVecsPerKBlock;

#pragma unroll
  for (int iter = 0; iter < (kMaxTotalChunks + kThreads - 1) / kThreads; ++iter) {
    const int chunk_idx = threadIdx.x + iter * kThreads;
    if (chunk_idx >= total_chunks) continue;
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

#pragma unroll
  for (int iter = 0; iter < (kStep1HiddenGroupWords * kStep1RowTile + kThreads - 1) / kThreads;
       ++iter) {
    const int linear_idx = threadIdx.x + iter * kThreads;
    if (linear_idx >= total_words) continue;
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
  constexpr int kMaxTotalWords = kStep1TmaRawChunkBlocks * kWordsPerKBlock * kStep1RowTile;
  const int total_words = k_blk_count * kWordsPerKBlock * kStep1RowTile;

#pragma unroll
  for (int iter = 0; iter < (kMaxTotalWords + kThreads - 1) / kThreads; ++iter) {
    const int linear_idx = threadIdx.x + iter * kThreads;
    if (linear_idx >= total_words) continue;
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

template <int kThreads>
__device__ __forceinline__ void load_hidden_wordtiles_group_sw128(
    uint8_t* smem_B_bytes,
    const uint4* __restrict__ hidden_vec_dev,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int row_tile,
    int n_rows,
    int k_blk_base,
    int k_blk_count) {
  constexpr int kHiddenVecs = kStep1Hidden / 16;
  constexpr int kVecsPerKBlock = kStep1Block / 16;
  constexpr int kMaxTotalVecs = kStep1TmaRawChunkBlocks * kStep1RowTile * kVecsPerKBlock;
  const int total_vecs = k_blk_count * kStep1RowTile * kVecsPerKBlock;

#pragma unroll
  for (int iter = 0; iter < (kMaxTotalVecs + kThreads - 1) / kThreads; ++iter) {
    const int linear_idx = threadIdx.x + iter * kThreads;
    if (linear_idx >= total_vecs) continue;
    const int vecs_per_block = kStep1RowTile * kVecsPerKBlock;
    const int blk_local = linear_idx / vecs_per_block;
    const int block_offset = linear_idx - blk_local * vecs_per_block;
    const int rr = block_offset / kVecsPerKBlock;
    const int chunk16 = block_offset - rr * kVecsPerKBlock;

    uint4 value = make_uint4(0u, 0u, 0u, 0u);
    if (rr < n_rows) {
      const int packed_idx = packed_base + row_tile + rr;
      const int token_idx = valid_token_idx[packed_idx];
      const int k_blk = k_blk_base + blk_local;
      const int k0_vec = (k_blk * kStep1Block) / 16;
      const int vec_idx = static_cast<int64_t>(token_idx) * kHiddenVecs + k0_vec + chunk16;
      value = hidden_vec_dev[vec_idx];
    }

    const int phys_chunk16 = chunk16 ^ rr;
    const int dst_byte =
        blk_local * kStep1TcgenBCompactBytes + rr * kStep1Block + phys_chunk16 * 16;
    *reinterpret_cast<uint4*>(smem_B_bytes + dst_byte) = value;
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
    uint64_t* barrier,
    int gate_up_tiles = 2) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (w13_tma_desc == nullptr) return false;
  const uint32_t kBytesPerChunk =
      gate_up_tiles * out_rows * kStep1Block * h_tiles;
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
  (void)gate_up_tiles;
  return false;
#endif
}

__device__ __forceinline__ bool issue_weight_tma_sw128_64x128(
    uint8_t* smem_weight_tile,
    const void* __restrict__ w13_tma_desc,
    int expert,
    int out_row_base,
    int k_blk,
    uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (w13_tma_desc == nullptr) return false;
  constexpr uint32_t kBytesPerTile =
      static_cast<uint32_t>(kStep1TcgenM) * static_cast<uint32_t>(kStep1Block);
  if (threadIdx.x == 0) {
    const int32_t coords[5] = {
        0,
        out_row_base,
        0,
        k_blk,
        expert};
    cuda::ptx::mbarrier_init(barrier, 1);
    cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global, cuda::ptx::cta_group_1,
        smem_weight_tile, w13_tma_desc, coords, barrier);
    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared,
        barrier, kBytesPerTile);
  }
  return true;
#else
  (void)smem_weight_tile;
  (void)w13_tma_desc;
  (void)expert;
  (void)out_row_base;
  (void)k_blk;
  (void)barrier;
  return false;
#endif
}

__device__ __forceinline__ bool issue_weight_tma_sw128_64x1024(
    uint8_t* smem_weight_group,
    const void* __restrict__ w13_tma_desc,
    int expert,
    int out_row_base,
    int k_group,
    uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (w13_tma_desc == nullptr) return false;
  constexpr uint32_t kBytesPerGroup =
      static_cast<uint32_t>(kStep1TcgenM) *
      static_cast<uint32_t>(kStep1Block) *
      static_cast<uint32_t>(kStep1TmaRawChunkBlocks);
  if (threadIdx.x == 0) {
    const int32_t coords[5] = {
        0,
        out_row_base,
        0,
        k_group * kStep1TmaRawChunkBlocks,
        expert};
    cuda::ptx::mbarrier_init(barrier, 1);
    cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global, cuda::ptx::cta_group_1,
        smem_weight_group, w13_tma_desc, coords, barrier);
    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared,
        barrier, kBytesPerGroup);
  }
  return true;
#else
  (void)smem_weight_group;
  (void)w13_tma_desc;
  (void)expert;
  (void)out_row_base;
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

__device__ __forceinline__ constexpr uint32_t make_tcgen05_idesc_f8f6f4_f16_dense(int m, int n) {
  return (static_cast<uint32_t>(n >> 3) << 17) | (static_cast<uint32_t>(m >> 4) << 24);
}

__device__ __forceinline__ constexpr uint32_t make_tcgen05_idesc_f8f6f4_dense(
    int m, int n, int accum_mode) {
  return accum_mode == 1 ? make_tcgen05_idesc_f8f6f4_f16_dense(m, n)
                         : make_tcgen05_idesc_f8f6f4_f32_dense(m, n);
}

__device__ __forceinline__ constexpr uint32_t make_tcgen05_idesc_tf32_ts_f32_dense(int m, int n) {
  return (1u << 4) | (2u << 7) | (2u << 10) |
         (static_cast<uint32_t>(n >> 3) << 17) |
         (static_cast<uint32_t>(m >> 4) << 24);
}

template <int AccumMode>
__device__ __forceinline__ constexpr uint32_t make_tcgen05_idesc_f8f6f4_dense(int m, int n) {
  if constexpr (AccumMode == 1) {
    return make_tcgen05_idesc_f8f6f4_f16_dense(m, n);
  } else {
    return make_tcgen05_idesc_f8f6f4_f32_dense(m, n);
  }
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

__device__ __forceinline__ void tcgen05_mma_tf32_cta1_ts(
    uint32_t d_tmem,
    uint32_t a_tmem,
    uint64_t b_desc,
    uint32_t idesc,
    bool enable_input_d) {
  uint32_t enable_u32 = enable_input_d ? 1u : 0u;
  uint32_t disable_output_lane[4] = {0, 0, 0, 0};
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.u32 p, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, {%4, %5, %6, %7}, p;\n\t"
      "}\n"
      :
      : "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc),
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

__device__ __forceinline__ float tcgen05_ld_16x64b_x1_to_f32(uint32_t taddr) {
  uint32_t bits;
  asm volatile("tcgen05.ld.sync.aligned.16x64b.x1.b32 "
               "{%0}, [%1];"
               : "=r"(bits)
               : "r"(taddr)
               : "memory");
  return __uint_as_float(bits);
}

__device__ __forceinline__ float tcgen05_ld_16x64b_x1_f16_to_f32(uint32_t taddr) {
  uint32_t bits;
  asm volatile("tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 "
               "{%0}, [%1];"
               : "=r"(bits)
               : "r"(taddr)
               : "memory");
  return __half2float(__ushort_as_half(static_cast<unsigned short>(bits & 0xffffu)));
}

__device__ __forceinline__ float tcgen05_ld_16x64b_x1_accum_to_f32(
    uint32_t taddr, int accum_mode) {
  return accum_mode == 1 ? tcgen05_ld_16x64b_x1_f16_to_f32(taddr)
                         : tcgen05_ld_16x64b_x1_to_f32(taddr);
}

__device__ __forceinline__ void tcgen05_ld_16x64b_x4_f16_to_f32(
    float (&dst)[4], uint32_t taddr, int col_parity) {
  uint32_t bits[4];
  asm volatile("tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 "
               "{%0, %1, %2, %3}, [%4];"
               : "=r"(bits[0]), "=r"(bits[1]), "=r"(bits[2]), "=r"(bits[3])
               : "r"(taddr)
               : "memory");
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned half_bits =
        col_parity == 0 ? (bits[i] & 0xffffu) : ((bits[i] >> 16) & 0xffffu);
    dst[i] = __half2float(__ushort_as_half(static_cast<unsigned short>(half_bits)));
  }
}

__device__ __forceinline__ void tcgen05_ld_16x64b_x4_accum_to_f32(
    float (&dst)[4], uint32_t taddr, int accum_mode, int col_parity) {
  if (accum_mode == 1) {
    tcgen05_ld_16x64b_x4_f16_to_f32(dst, taddr, col_parity);
  } else {
    tcgen05_ld_16x64b_x4(dst, taddr);
  }
}

template <int AccumMode>
__device__ __forceinline__ void tcgen05_ld_16x64b_x4_accum_to_f32(
    float (&dst)[4], uint32_t taddr, int col_parity) {
  if constexpr (AccumMode == 1) {
    tcgen05_ld_16x64b_x4_f16_to_f32(dst, taddr, col_parity);
  } else {
    tcgen05_ld_16x64b_x4(dst, taddr);
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

__device__ __forceinline__ uint32_t tcgen05_alloc_cta1_cols_64(uint32_t* smem_out_taddr) {
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_out_taddr);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"
               :: "r"(smem_addr)
               : "memory");
  __syncwarp();
  uint32_t taddr;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr) : "r"(smem_addr) : "memory");
  return taddr;
}

__device__ __forceinline__ void tcgen05_dealloc_cta1_cols_64(uint32_t taddr) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;"
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

#pragma unroll
  for (int iter = 0; iter < (kAStageBytes + kStep1CommThreads - 1) / kStep1CommThreads; ++iter) {
    const int linear = threadIdx.x + iter * kStep1CommThreads;
    if (linear >= kAStageBytes) continue;
    const int m = linear / kStep1TcgenCompactKBytes;
    const int k = linear - m * kStep1TcgenCompactKBytes;
    const int row_group = m >> 3;
    const int row_in8 = m & 7;
    const int slice = k >> 4;
    const int byte = k & 15;
    const int src = (m < kStep1OutRowsPerCta)
                        ? (m * kStep1HiddenGroupBytes + byte_base + k)
                        : (up_group_base + (m - kStep1OutRowsPerCta) * kStep1HiddenGroupBytes +
                           byte_base + k);
    smem_A_compact[(row_group * 16 + slice * 8 + row_in8) * 16 + byte] =
        smem_A_bytes[src];
  }

#pragma unroll
  for (int iter = 0; iter < (kBStageBytes + kStep1CommThreads - 1) / kStep1CommThreads; ++iter) {
    const int linear = threadIdx.x + iter * kStep1CommThreads;
    if (linear >= kBStageBytes) continue;
    const int n = linear / kStep1TcgenCompactKBytes;
    const int k = linear - n * kStep1TcgenCompactKBytes;
    const int slice = k >> 4;
    const int byte = k & 15;
    smem_B_compact[(slice * 8 + n) * 16 + byte] =
        smem_B_bytes[n * kStep1HiddenGroupBytes + byte_base + k];
  }
}

template <class TensorA, class TensorB>
__device__ __forceinline__ void stage_tcgen05_compact_ab_64x8x128(
    TensorA smem_A_compact,
    TensorB smem_B_compact,
    const uint32_t* __restrict__ smem_A_group_combined_words,
    const uint32_t* __restrict__ smem_B_group_words,
    int k_blk_local) {
  const uint8_t* smem_A_bytes = reinterpret_cast<const uint8_t*>(smem_A_group_combined_words);
  const uint8_t* smem_B_bytes = reinterpret_cast<const uint8_t*>(smem_B_group_words);
  const int byte_base = k_blk_local * kStep1Block;
  const int up_group_base = kStep1OutRowsPerCta * kStep1HiddenGroupBytes;
  constexpr int kAStageBytes = kStep1TcgenM * kStep1Block;
  constexpr int kBStageBytes = kStep1TcgenN * kStep1Block;

#pragma unroll
  for (int iter = 0; iter < (kAStageBytes + kStep1CommThreads - 1) / kStep1CommThreads; ++iter) {
    const int linear = threadIdx.x + iter * kStep1CommThreads;
    if (linear >= kAStageBytes) continue;
    const int m = linear / kStep1Block;
    const int k = linear - m * kStep1Block;
    const int row_group = m >> 3;
    const int row_in8 = m & 7;
    const int chunk16 = k >> 4;
    const int byte = k & 15;
    const int phys_chunk16 = chunk16 ^ row_in8;
    const int src = (m < kStep1OutRowsPerCta)
                        ? (m * kStep1HiddenGroupBytes + byte_base + k)
                        : (up_group_base + (m - kStep1OutRowsPerCta) * kStep1HiddenGroupBytes +
                           byte_base + k);
    smem_A_compact[row_group * 8 * kStep1Block + row_in8 * kStep1Block +
                   phys_chunk16 * 16 + byte] =
        smem_A_bytes[src];
  }

#pragma unroll
  for (int iter = 0; iter < (kBStageBytes + kStep1CommThreads - 1) / kStep1CommThreads; ++iter) {
    const int linear = threadIdx.x + iter * kStep1CommThreads;
    if (linear >= kBStageBytes) continue;
    const int n = linear / kStep1Block;
    const int k = linear - n * kStep1Block;
    const int chunk16 = k >> 4;
    const int byte = k & 15;
    const int phys_chunk16 = chunk16 ^ n;
    smem_B_compact[n * kStep1Block + phys_chunk16 * 16 + byte] =
        smem_B_bytes[n * kStep1HiddenGroupBytes + byte_base + k];
  }
}

template <class TensorA>
__device__ __forceinline__ void stage_tcgen05_compact_a_64x128(
    TensorA smem_A_compact,
    const uint32_t* __restrict__ smem_A_group_combined_words,
    int k_blk_local) {
  const uint8_t* smem_A_bytes = reinterpret_cast<const uint8_t*>(smem_A_group_combined_words);
  const int byte_base = k_blk_local * kStep1Block;
  const int up_group_base = kStep1OutRowsPerCta * kStep1HiddenGroupBytes;
  constexpr int kAStageBytes = kStep1TcgenM * kStep1Block;

#pragma unroll
  for (int iter = 0; iter < (kAStageBytes + kStep1CommThreads - 1) / kStep1CommThreads; ++iter) {
    const int linear = threadIdx.x + iter * kStep1CommThreads;
    if (linear >= kAStageBytes) continue;
    const int m = linear / kStep1Block;
    const int k = linear - m * kStep1Block;
    const int row_group = m >> 3;
    const int row_in8 = m & 7;
    const int chunk16 = k >> 4;
    const int byte = k & 15;
    const int phys_chunk16 = chunk16 ^ row_in8;
    const int src = (m < kStep1OutRowsPerCta)
                        ? (m * kStep1HiddenGroupBytes + byte_base + k)
                        : (up_group_base + (m - kStep1OutRowsPerCta) * kStep1HiddenGroupBytes +
                           byte_base + k);
    smem_A_compact[row_group * 8 * kStep1Block + row_in8 * kStep1Block +
                   phys_chunk16 * 16 + byte] =
        smem_A_bytes[src];
  }
}

template <class TensorB>
__device__ __forceinline__ void stage_tcgen05_compact_b_8x128(
    TensorB smem_B_compact,
    const uint32_t* __restrict__ smem_B_group_words,
    int k_blk_local) {
  const uint8_t* smem_B_bytes = reinterpret_cast<const uint8_t*>(smem_B_group_words);
  const int byte_base = k_blk_local * kStep1Block;
  constexpr int kBStageBytes = kStep1TcgenN * kStep1Block;

#pragma unroll
  for (int iter = 0; iter < (kBStageBytes + kStep1CommThreads - 1) / kStep1CommThreads; ++iter) {
    const int linear = threadIdx.x + iter * kStep1CommThreads;
    if (linear >= kBStageBytes) continue;
    const int n = linear / kStep1Block;
    const int k = linear - n * kStep1Block;
    const int chunk16 = k >> 4;
    const int byte = k & 15;
    const int phys_chunk16 = chunk16 ^ n;
    smem_B_compact[n * kStep1Block + phys_chunk16 * 16 + byte] =
        smem_B_bytes[n * kStep1HiddenGroupBytes + byte_base + k];
  }
}

__device__ __forceinline__ uint64_t make_tcgen05_core_matrix_desc_group8(const void* smem) {
  const uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem);
  const uint32_t matrix_start_aligned = smem_addr & ~0xFu;
  const uint32_t lbo = 8u * 16u;
  const uint32_t sbo = 16u * 16u;
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>(matrix_start_aligned >> 4);
  desc |= static_cast<uint64_t>((lbo & 0x3ffffu) >> 4) << 16;
  desc |= static_cast<uint64_t>((sbo & 0x3ffffu) >> 4) << 32;
  desc |= static_cast<uint64_t>(1u) << 46;
  desc |= static_cast<uint64_t>(0xb0u) << 53;
  return desc;
}

__device__ __forceinline__ uint64_t make_tcgen05_core_matrix_desc_group8_k128(const void* smem) {
  const uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem);
  const uint32_t matrix_start_aligned = smem_addr & ~0xFu;
  const uint32_t lbo = 16u;
  const uint32_t sbo = 8u * static_cast<uint32_t>(kStep1Block);
  const uint32_t boundary = 1024u;
  const uint32_t pattern_start_addr = matrix_start_aligned - (matrix_start_aligned % boundary);
  const uint32_t base_offset = (pattern_start_addr >> 7) & 0x7u;
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>(matrix_start_aligned >> 4);
  desc |= static_cast<uint64_t>((lbo & 0x3ffffu) >> 4) << 16;
  desc |= static_cast<uint64_t>((sbo & 0x3ffffu) >> 4) << 32;
  desc |= static_cast<uint64_t>(1u) << 46;
  desc |= static_cast<uint64_t>(base_offset & 0x7u) << 49;
  desc |= static_cast<uint64_t>(0xb0u) << 53;
  desc |= static_cast<uint64_t>(kStep1TcgenLayoutSwizzle128B & 0x7u) << 61;
  return desc;
}

__device__ __forceinline__ uint64_t make_tcgen05_tf32_diag_desc(const void* smem) {
  const uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem);
  const uint32_t matrix_start_aligned = smem_addr & ~0xFu;
  const uint32_t lbo = kStep1TcgenN * 16u;
  const uint32_t sbo = kStep1TcgenN * 16u;
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>(matrix_start_aligned >> 4);
  desc |= static_cast<uint64_t>((lbo & 0x3ffffu) >> 4) << 16;
  desc |= static_cast<uint64_t>((sbo & 0x3ffffu) >> 4) << 32;
  desc |= static_cast<uint64_t>(1u) << 46;
  desc |= static_cast<uint64_t>(0xb0u) << 53;
  return desc;
}

__device__ __forceinline__ float tf32_rne_prebias(float value) {
  const uint32_t bits = __float_as_uint(value);
  // const uint32_t exp = bits & 0x7f800000u;
  // if ((bits & 0x7fffffffu) == 0u || exp == 0x7f800000u) {
  //   return value;
  // }
  return __uint_as_float(bits + 0x1000u);
}

__device__ __forceinline__ void stage_tcgen05_tf32_diag_scale(
    float* smem_diag,
    float block_scale,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int row_tile,
    int n_rows,
    int k_blk) {
  constexpr int kDiagElems = 2 * kStep1TcgenN * 4;
#pragma unroll
  for (int iter = 0; iter < (kDiagElems + kStep1CommThreads - 1) / kStep1CommThreads; ++iter) {
    const int linear = threadIdx.x + iter * kStep1CommThreads;
    if (linear >= kDiagElems) continue;
    const int slice = linear / (kStep1TcgenN * 4);
    const int rem = linear - slice * kStep1TcgenN * 4;
    const int n = rem / 4;
    const int k_in_slice = rem - n * 4;
    const int k = slice * 4 + k_in_slice;
    float value = 0.0f;
    if (n == k && n < n_rows) {
      const int packed_idx = packed_base + row_tile + n;
      const int token_idx = valid_token_idx[packed_idx];
      value = block_scale * hidden_scale_dev[static_cast<int64_t>(k_blk) * t + token_idx];
    }
    smem_diag[linear] = tf32_rne_prebias(value);
  }
}

__device__ __forceinline__ void zero_tcgen05_tf32_scale_buffers(float* smem_diag) {
  constexpr int kScaleElems = kStep1TcgenScaleDiagBytes / sizeof(float);
#pragma unroll
  for (int iter = 0; iter < (kScaleElems + kStep1CommThreads - 1) / kStep1CommThreads; ++iter) {
    const int linear = threadIdx.x + iter * kStep1CommThreads;
    if (linear < kScaleElems) {
      smem_diag[linear] = 0.0f;
    }
  }
}

__device__ __forceinline__ void stage_tcgen05_tf32_all_diag_scales(
    float* smem_diag,
    const float* __restrict__ s13_all_dev,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ valid_token_idx,
    int expert,
    int out_block128,
    int packed_base,
    int row_tile,
    int n_rows) {
  constexpr int kScaleElems = kStep1TcgenScaleDiagBytes / sizeof(float);
#pragma unroll
  for (int iter = 0; iter < (kScaleElems + kStep1CommThreads - 1) / kStep1CommThreads; ++iter) {
    const int linear = threadIdx.x + iter * kStep1CommThreads;
    if (linear >= kScaleElems) continue;
    const int elems_per_matrix = 2 * kStep1TcgenN * 4;
    const int matrix = linear / elems_per_matrix;
    const int rem = linear - matrix * elems_per_matrix;
    const int k_blk = matrix >> 1;
    const int is_up = matrix & 1;
    const int slice = rem / (kStep1TcgenN * 4);
    const int rem2 = rem - slice * kStep1TcgenN * 4;
    const int n = rem2 / 4;
    const int k_in_slice = rem2 - n * 4;
    const int logical_k = slice * 4 + k_in_slice;
    float value = 0.0f;
    if (n == logical_k && n < n_rows) {
      const int packed_idx = packed_base + row_tile + n;
      const int token_idx = valid_token_idx[packed_idx];
      const int scale_block =
          is_up ? (out_block128 + kStep1IntermediateBlocks) : out_block128;
      const float block_scale =
          s13_all_dev[(expert * kStep1Gemm1OutBlocks + scale_block) *
                          kStep1HiddenBlocks +
                      k_blk];
      value = block_scale * hidden_scale_dev[static_cast<int64_t>(k_blk) * t + token_idx];
    }
    smem_diag[linear] = tf32_rne_prebias(value);
  }
}

__device__ __forceinline__ void update_tcgen05_tf32_all_diag_values(
    float* smem_diag,
    const float* __restrict__ s13_all_dev,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ valid_token_idx,
    int expert,
    int out_block128,
    int packed_base,
    int row_tile,
    int n_rows) {
  constexpr int kDiagValues = 2 * kStep1HiddenBlocks * kStep1TcgenN;
#pragma unroll
  for (int iter = 0; iter < (kDiagValues + kStep1CommThreads - 1) / kStep1CommThreads; ++iter) {
    const int linear = threadIdx.x + iter * kStep1CommThreads;
    if (linear >= kDiagValues) continue;
    const int matrix = linear / kStep1TcgenN;
    const int n = linear - matrix * kStep1TcgenN;
    const int k_blk = matrix >> 1;
    const int is_up = matrix & 1;
    float value = 0.0f;
    if (n < n_rows) {
      const int packed_idx = packed_base + row_tile + n;
      const int token_idx = valid_token_idx[packed_idx];
      const int scale_block =
          is_up ? (out_block128 + kStep1IntermediateBlocks) : out_block128;
      const float block_scale =
          s13_all_dev[(expert * kStep1Gemm1OutBlocks + scale_block) *
                          kStep1HiddenBlocks +
                      k_blk];
      value = block_scale * hidden_scale_dev[static_cast<int64_t>(k_blk) * t + token_idx];
    }
    const int slice = n >> 2;
    const int k_in_slice = n & 3;
    smem_diag[matrix * (2 * kStep1TcgenN * 4) +
              slice * kStep1TcgenN * 4 + n * 4 + k_in_slice] = tf32_rne_prebias(value);
  }
}

__device__ __forceinline__ void update_tcgen05_tf32_diag_only(
    float* smem_diag,
    float block_scale,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int row_tile,
    int n_rows,
    int k_blk) {
  if (threadIdx.x < kStep1TcgenN) {
    const int n = threadIdx.x;
    float value = 0.0f;
    if (n < n_rows) {
      const int packed_idx = packed_base + row_tile + n;
      const int token_idx = valid_token_idx[packed_idx];
      value = block_scale * hidden_scale_dev[static_cast<int64_t>(k_blk) * t + token_idx];
    }
    const int slice = n >> 2;
    const int k_in_slice = n & 3;
    smem_diag[slice * kStep1TcgenN * 4 + n * 4 + k_in_slice] = tf32_rne_prebias(value);
  }
}
#else
__device__ __forceinline__ void zero_tcgen05_tf32_scale_buffers(float* smem_diag) {
  (void)smem_diag;
}

__device__ __forceinline__ void stage_tcgen05_tf32_all_diag_scales(
    float* smem_diag,
    const float* __restrict__ s13_all_dev,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ valid_token_idx,
    int expert,
    int out_block128,
    int packed_base,
    int row_tile,
    int n_rows) {
  (void)smem_diag;
  (void)s13_all_dev;
  (void)hidden_scale_dev;
  (void)t;
  (void)valid_token_idx;
  (void)expert;
  (void)out_block128;
  (void)packed_base;
  (void)row_tile;
  (void)n_rows;
}

__device__ __forceinline__ void update_tcgen05_tf32_all_diag_values(
    float* smem_diag,
    const float* __restrict__ s13_all_dev,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ valid_token_idx,
    int expert,
    int out_block128,
    int packed_base,
    int row_tile,
    int n_rows) {
  (void)smem_diag;
  (void)s13_all_dev;
  (void)hidden_scale_dev;
  (void)t;
  (void)valid_token_idx;
  (void)expert;
  (void)out_block128;
  (void)packed_base;
  (void)row_tile;
  (void)n_rows;
}

__device__ __forceinline__ void update_tcgen05_tf32_diag_only(
    float* smem_diag,
    float block_scale,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int row_tile,
    int n_rows,
    int k_blk) {
  (void)smem_diag;
  (void)block_scale;
  (void)hidden_scale_dev;
  (void)t;
  (void)valid_token_idx;
  (void)packed_base;
  (void)row_tile;
  (void)n_rows;
  (void)k_blk;
}
#endif

template <int AccumMode>
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
    int& mma_phase_bit,
    bool a_already_sw128,
    bool b_already_sw128) {
#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) && \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t result_tmem = tmem_base_ptr;
  const uint32_t idesc =
      make_tcgen05_idesc_f8f6f4_dense<AccumMode>(kStep1TcgenM, kStep1TcgenN);
  const uint64_t a_desc = make_tcgen05_core_matrix_desc_group8(smem_A_tcgen_bytes);
  const uint64_t b_desc = make_tcgen05_core_matrix_desc_group8(smem_B_tcgen_bytes);

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
      const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
      const int col_parity = (lane >> 1) & 1;
      tcgen05_ld_16x64b_x4_accum_to_f32<AccumMode>(
          result_bits, result_tmem, col_parity);
      tcgen05_wait_ld_sync();
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
  (void)a_already_sw128;
  (void)b_already_sw128;
#endif
}

template <int AccumMode>
__device__ __forceinline__ void mma_64x8x128_tcgen05(
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
    int lane,
    int warp_id,
    int debug_output_mode,
    uint32_t tmem_base_ptr,
    uint64_t* mma_barrier,
    int& mma_phase_bit,
    bool a_already_sw128,
    bool b_already_sw128) {
#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) && \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t result_tmem = tmem_base_ptr;
  const uint32_t idesc =
      make_tcgen05_idesc_f8f6f4_dense<AccumMode>(kStep1TcgenM, kStep1TcgenN);

  if (a_already_sw128 && b_already_sw128) {
    // A is already in smem_A_tcgen and B is already in swizzled 8x128 tiles.
  } else if (a_already_sw128) {
    stage_tcgen05_compact_b_8x128(smem_B_tcgen_bytes, smem_B_group_words, k_blk_local);
  } else if (b_already_sw128) {
    stage_tcgen05_compact_a_64x128(
        smem_A_tcgen_bytes, smem_A_group_combined_words, k_blk_local);
  } else {
    stage_tcgen05_compact_ab_64x8x128(
        smem_A_tcgen_bytes, smem_B_tcgen_bytes, smem_A_group_combined_words,
        smem_B_group_words, k_blk_local);
  }
  __syncthreads();

  const uint8_t* a_base = smem_A_tcgen_bytes;
  const uint8_t* b_base =
      b_already_sw128 ? reinterpret_cast<const uint8_t*>(smem_B_group_words) +
                            static_cast<int64_t>(k_blk_local) * kStep1TcgenBCompactBytes
                      : smem_B_tcgen_bytes;

#pragma unroll
  for (int issue = 0; issue < kStep1KSubsPerBlock; ++issue) {
    const int slice_base = issue * (kStep1TcgenCompactKBytes / 16);
    const uint8_t* a_issue = a_base + slice_base * 16;
    const uint8_t* b_issue = b_base + slice_base * 16;
    const uint64_t a_desc = make_tcgen05_core_matrix_desc_group8_k128(a_issue);
    const uint64_t b_desc = make_tcgen05_core_matrix_desc_group8_k128(b_issue);
    if (warp_id == 0 && lane == 0) {
      tcgen05_mma_f8f6f4_cta1_ss(result_tmem, a_desc, b_desc, idesc, issue != 0);
    }
  }
  if (warp_id == 0 && lane == 0) {
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

  if (warp_id < 4) {
    float result_bits[4];
    const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int col_parity = (lane >> 1) & 1;
    tcgen05_ld_16x64b_x4_accum_to_f32<AccumMode>(
        result_bits, result_tmem, col_parity);
    tcgen05_wait_ld_sync();
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
  (void)lane;
  (void)warp_id;
  (void)debug_output_mode;
  (void)tmem_base_ptr;
  (void)mma_barrier;
  (void)mma_phase_bit;
  (void)a_already_sw128;
  (void)b_already_sw128;
#endif
}

__device__ __forceinline__ void mma_64x8x128_tcgen05(
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
    int lane,
    int warp_id,
    int debug_output_mode,
    uint32_t tmem_base_ptr,
    uint64_t* mma_barrier,
    int& mma_phase_bit,
    bool a_already_sw128,
    bool b_already_sw128,
    int tcgen_accum_mode) {
  if (tcgen_accum_mode == 1) {
    mma_64x8x128_tcgen05<1>(
        gate_acc, up_acc, smem_A_tcgen_bytes, smem_B_tcgen_bytes,
        smem_A_group_combined_words, smem_B_group_words, gate_block_scale, up_block_scale,
        hidden_scale_dev, t, valid_token_idx, packed_base, row_tile, n_rows, k_blk,
        k_blk_local, lane, warp_id, debug_output_mode, tmem_base_ptr, mma_barrier,
        mma_phase_bit, a_already_sw128, b_already_sw128);
  } else {
    mma_64x8x128_tcgen05<0>(
        gate_acc, up_acc, smem_A_tcgen_bytes, smem_B_tcgen_bytes,
        smem_A_group_combined_words, smem_B_group_words, gate_block_scale, up_block_scale,
        hidden_scale_dev, t, valid_token_idx, packed_base, row_tile, n_rows, k_blk,
        k_blk_local, lane, warp_id, debug_output_mode, tmem_base_ptr, mma_barrier,
        mma_phase_bit, a_already_sw128, b_already_sw128);
  }
}

template <int AccumMode>
__device__ __forceinline__ void issue_mma_64x8x128_tcgen05_no_wait(
    uint8_t* smem_A_tcgen_bytes,
    const uint32_t* __restrict__ smem_B_group_words,
    int k_blk_local,
    int lane,
    int warp_id,
    uint32_t tmem_base_ptr,
    bool enable_input_d) {
#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) && \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t idesc =
      make_tcgen05_idesc_f8f6f4_dense<AccumMode>(kStep1TcgenM, kStep1TcgenN);
  const uint8_t* a_base = smem_A_tcgen_bytes;
  const uint8_t* b_base =
      reinterpret_cast<const uint8_t*>(smem_B_group_words) +
      static_cast<int64_t>(k_blk_local) * kStep1TcgenBCompactBytes;
#pragma unroll
  for (int issue = 0; issue < kStep1KSubsPerBlock; ++issue) {
    const int slice_base = issue * (kStep1TcgenCompactKBytes / 16);
    const uint8_t* a_issue = a_base + slice_base * 16;
    const uint8_t* b_issue = b_base + slice_base * 16;
    const uint64_t a_desc = make_tcgen05_core_matrix_desc_group8_k128(a_issue);
    const uint64_t b_desc = make_tcgen05_core_matrix_desc_group8_k128(b_issue);
    if (warp_id == 0 && lane == 0) {
      tcgen05_mma_f8f6f4_cta1_ss(
          tmem_base_ptr, a_desc, b_desc, idesc, enable_input_d || (issue != 0));
    }
  }
#else
  (void)smem_A_tcgen_bytes;
  (void)smem_B_group_words;
  (void)k_blk_local;
  (void)lane;
  (void)warp_id;
  (void)tmem_base_ptr;
  (void)enable_input_d;
#endif
}

__device__ __forceinline__ void issue_mma_64x8x128_tcgen05_no_wait(
    uint8_t* smem_A_tcgen_bytes,
    const uint32_t* __restrict__ smem_B_group_words,
    int k_blk_local,
    int lane,
    int warp_id,
    uint32_t tmem_base_ptr,
    bool enable_input_d,
    int tcgen_accum_mode) {
  if (tcgen_accum_mode == 1) {
    issue_mma_64x8x128_tcgen05_no_wait<1>(
        smem_A_tcgen_bytes, smem_B_group_words, k_blk_local, lane, warp_id,
        tmem_base_ptr, enable_input_d);
  } else {
    issue_mma_64x8x128_tcgen05_no_wait<0>(
        smem_A_tcgen_bytes, smem_B_group_words, k_blk_local, lane, warp_id,
        tmem_base_ptr, enable_input_d);
  }
}

__device__ __forceinline__ void read_scale_accumulate_tcgen05_group_8slots(
    LaneAccum8x1& gate_acc,
    LaneAccum8x1& up_acc,
    const float* __restrict__ s13_all_dev,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int row_tile,
    int n_rows,
    int expert,
    int out_block128,
    int k_group,
    int lane,
    int warp_id,
    int debug_output_mode,
    uint32_t tmem_group_base,
    int tcgen_accum_mode) {
#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (warp_id < 4) {
    const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
    const int col_parity = (lane >> 1) & 1;
#pragma unroll
    for (int k_blk_local = 0; k_blk_local < kStep1TmaRawChunkBlocks; ++k_blk_local) {
      const int k_blk = k_group * kStep1TmaRawChunkBlocks + k_blk_local;
      float result_bits[4];
      tcgen05_ld_16x64b_x4_accum_to_f32(
          result_bits,
          tmem_group_base + static_cast<uint32_t>(k_blk_local * kStep1TcgenN),
          tcgen_accum_mode,
          col_parity);
      tcgen05_wait_ld_sync();
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
          const float gate_scale =
              s13_all_dev[(expert * kStep1Gemm1OutBlocks + out_block128) *
                              kStep1HiddenBlocks +
                          k_blk];
          const float scale = dump_unscaled_gate ? 1.0f : gate_scale * hidden_block_scale;
          gate_acc.v[rr] += result_bits[reg] * scale;
        } else {
          const float up_scale =
              s13_all_dev[(expert * kStep1Gemm1OutBlocks +
                           (out_block128 + kStep1IntermediateBlocks)) *
                              kStep1HiddenBlocks +
                          k_blk];
          const float scale = dump_unscaled_up ? 1.0f : up_scale * hidden_block_scale;
          up_acc.v[rr] += result_bits[reg] * scale;
        }
      }
    }
  }
#else
  (void)gate_acc;
  (void)up_acc;
  (void)s13_all_dev;
  (void)hidden_scale_dev;
  (void)t;
  (void)valid_token_idx;
  (void)packed_base;
  (void)row_tile;
  (void)n_rows;
  (void)expert;
  (void)out_block128;
  (void)k_group;
  (void)lane;
  (void)warp_id;
  (void)debug_output_mode;
  (void)tmem_group_base;
  (void)tcgen_accum_mode;
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

template <int kFastTcgenAccumMode>
static __global__ __launch_bounds__(kStep1CommThreads, 2) void step1_gemm1_swiglu_direct_kernel(
    const uint8_t* __restrict__ hidden_fp8_dev,
    const float* __restrict__ hidden_scale_dev,
    int64_t t,
    const int* __restrict__ expert_t_valid,
    const int* __restrict__ expert_offset,
    const int* __restrict__ valid_token_idx,
    const int* __restrict__ active_experts,
    const uint8_t* __restrict__ w13_all_dev,
    const float* __restrict__ s13_all_dev,
    const void* __restrict__ hidden_tma_desc,
    const void* __restrict__ w13_tma_desc,
    bool direct_tma_sw128,
    bool direct_b_sw128,
    int debug_output_mode,
    int tcgen_accum_mode,
    float* __restrict__ c_perm_all_dev) {
  (void)hidden_tma_desc;

  const int out_tile32 = blockIdx.x;  // 0..63
  const int expert = active_experts != nullptr ? active_experts[blockIdx.y] : blockIdx.y;
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
  const uint4* hidden_vec_dev = reinterpret_cast<const uint4*>(hidden_fp8_dev);

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t smem_raw_addr = cute::cast_smem_ptr_to_uint(smem_raw);
  const std::size_t smem_align_pad =
      (1024u - (smem_raw_addr & 1023u)) & 1023u;
  const std::size_t weight_group_bytes =
      static_cast<std::size_t>(2) * kStep1OutRowsPerCta * kStep1HiddenGroupBytes;
  const std::size_t hidden_group_offset =
      smem_align_pad + kStep1PipelineStages * weight_group_bytes;
  const std::size_t hidden_group_bytes =
      static_cast<std::size_t>(kStep1HiddenGroupBytes) * kStep1RowTile;
  const std::size_t tcgen_a_offset =
      align_up(hidden_group_offset + 2 * hidden_group_bytes, 1024);
  const std::size_t tcgen_b_offset = align_up(tcgen_a_offset + kStep1TcgenACompactBytes, 1024);
  const std::size_t tcgen_scale_offset =
      align_up(tcgen_b_offset + kStep1TcgenBCompactBytes, 1024);
  uint32_t* smem_A_group = reinterpret_cast<uint32_t*>(smem_raw + smem_align_pad);
  uint32_t* smem_A_group_alt =
      reinterpret_cast<uint32_t*>(smem_raw + smem_align_pad + weight_group_bytes);
  uint32_t* smem_B_group = reinterpret_cast<uint32_t*>(smem_raw + hidden_group_offset);
  uint32_t* smem_B_group_alt =
      reinterpret_cast<uint32_t*>(smem_raw + hidden_group_offset + hidden_group_bytes);
  uint8_t* smem_A_tcgen = smem_raw + tcgen_a_offset;
  uint8_t* smem_B_tcgen = smem_raw + tcgen_b_offset;
  float* smem_tcgen_scale = reinterpret_cast<float*>(smem_raw + tcgen_scale_offset);
  alignas(16) __shared__ float tcgen_up_acc_smem[kStep1ExchangeSmemElems];

#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  alignas(16) __shared__ uint64_t weight_group_barrier[kStep1PipelineStages];
  alignas(16) __shared__ uint64_t tcgen_mma_barrier;
  alignas(16) __shared__ uint32_t tcgen_tmem_base_ptr;
  if (threadIdx.x == 0) {
    tcgen05_mbarrier_init(&tcgen_mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_cta1_cols_64(&tcgen_tmem_base_ptr);
    if (lane == 0) {
      tcgen_tmem_base_ptr = taddr;
    }
  }
  __syncthreads();
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  alignas(16) __shared__ uint64_t weight_group_barrier[kStep1PipelineStages];
#endif

  for (int row_tile = 0; row_tile < t_valid; row_tile += kStep1RowTile) {
    const int n_rows = min_int(t_valid - row_tile, kStep1RowTile);

    const int acc_clear_idx = threadIdx.x;
    if (acc_clear_idx < kStep1ExchangeSmemElems) {
      tcgen_up_acc_smem[acc_clear_idx] = 0.0f;
    }
    __syncthreads();

    LaneAccum8x1 gate_acc;
    LaneAccum8x1 up_acc;
    if (warp_id < 4) {
      gate_acc.clear();
      up_acc.clear();
    }

    constexpr bool kFastTcgen = (kFastTcgenAccumMode >= 0);
    const int effective_tcgen_accum_mode = kFastTcgen ? kFastTcgenAccumMode :
        ((tcgen_accum_mode == 1 && t != 1) ? 0 : tcgen_accum_mode);
    const bool use_tma = (w13_tma_desc != nullptr);
    const bool use_direct_tma_sw128 = use_tma && direct_tma_sw128;
    const bool use_direct_b_sw128 = direct_b_sw128;
    const bool ablate_accum56 =
        (!kFastTcgen) &&
        (debug_output_mode == 98 || debug_output_mode == 99) &&
        use_direct_tma_sw128 && use_direct_b_sw128;
    const bool use_tmem_scale_mma = kFastTcgen ? true :
        (effective_tcgen_accum_mode == 2 || effective_tcgen_accum_mode == 3) &&
        (debug_output_mode == 0) && use_direct_tma_sw128 && use_direct_b_sw128;
    const bool ablate_tmem_scale_staging = effective_tcgen_accum_mode == 3;
    int tcgen_mma_phase_bit = 0;
    bool ablate_first_mma = true;

    if (ablate_accum56 || (use_direct_tma_sw128 && use_direct_b_sw128)) {
#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
      issue_weight_tma_sw128_64x1024(
          reinterpret_cast<uint8_t*>(smem_A_group), w13_tma_desc, expert,
          out_row_base, 0, &weight_group_barrier[0]);
      if (use_tmem_scale_mma && !ablate_tmem_scale_staging) {
        stage_tcgen05_tf32_all_diag_scales(
            smem_tcgen_scale, s13_all_dev, hidden_scale_dev, t, valid_token_idx,
            expert, out_block128, packed_base, row_tile, n_rows);
      }
      load_hidden_wordtiles_group_sw128<kStep1CommThreads>(
          reinterpret_cast<uint8_t*>(smem_B_group), hidden_vec_dev, valid_token_idx,
          packed_base, row_tile, n_rows, 0, kStep1TmaRawChunkBlocks);
      __syncthreads();
      wait_weight_group_tma_combined(&weight_group_barrier[0]);

      bool tmem_scale_first = true;
      const uint32_t tcgen_partial_tmem = tcgen_tmem_base_ptr;
      const uint32_t tcgen_gate_scaled_tmem =
          tcgen_tmem_base_ptr + kStep1TmaRawChunkBlocks * kStep1TcgenN;
      const uint32_t tcgen_up_scaled_tmem = tcgen_gate_scaled_tmem + kStep1TcgenN;
      for (int k_group = 0; k_group < kStep1TmaRawChunkGroups; ++k_group) {
        const int cur = k_group & 1;
        const int next = cur ^ 1;
        uint32_t* A_cur = cur ? smem_A_group_alt : smem_A_group;
        uint32_t* B_cur = cur ? smem_B_group_alt : smem_B_group;
        uint32_t* A_next = next ? smem_A_group_alt : smem_A_group;
        uint32_t* B_next = next ? smem_B_group_alt : smem_B_group;
        bool prefetched_next_hidden = false;

        if (k_group + 1 < kStep1TmaRawChunkGroups) {
          issue_weight_tma_sw128_64x1024(
              reinterpret_cast<uint8_t*>(A_next), w13_tma_desc, expert,
              out_row_base, k_group + 1, &weight_group_barrier[0]);
        }

        if (use_tmem_scale_mma) {
#pragma unroll
          for (int k_blk_local = 0; k_blk_local < kStep1TmaRawChunkBlocks; ++k_blk_local) {
            const int k_blk = k_group * kStep1TmaRawChunkBlocks + k_blk_local;
            uint8_t* smem_A_for_mma =
                reinterpret_cast<uint8_t*>(A_cur) +
                static_cast<int64_t>(k_blk_local) * kStep1TcgenACompactBytes;
            issue_mma_64x8x128_tcgen05_no_wait(
                smem_A_for_mma, B_cur, k_blk_local, lane, warp_id,
                tcgen_partial_tmem + static_cast<uint32_t>(k_blk_local * kStep1TcgenN),
                false, 0);

          }
          if (warp_id == 0 && lane == 0) {
            tcgen05_commit_group1(&tcgen_mma_barrier);
          }
          if (threadIdx.x == 0) {
            tcgen05_wait_mma_barrier_single(&tcgen_mma_barrier, tcgen_mma_phase_bit);
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          }
          __syncthreads();
          tcgen_mma_phase_bit ^= 1;

          const uint32_t ts_idesc =
              make_tcgen05_idesc_tf32_ts_f32_dense(kStep1TcgenM, kStep1TcgenN);
#pragma unroll
          for (int k_blk_local = 0; k_blk_local < kStep1TmaRawChunkBlocks; ++k_blk_local) {
            const int k_blk = k_group * kStep1TmaRawChunkBlocks + k_blk_local;
            const uint32_t partial_tmem =
                tcgen_partial_tmem + static_cast<uint32_t>(k_blk_local * kStep1TcgenN);
            const bool enable_acc = !tmem_scale_first || (k_blk_local != 0);
            float* gate_diag =
                smem_tcgen_scale +
                (2 * k_blk + 0) * (2 * kStep1TcgenN * 4);
            float* up_diag =
                smem_tcgen_scale +
                (2 * k_blk + 1) * (2 * kStep1TcgenN * 4);
            const uint64_t gate_desc = make_tcgen05_tf32_diag_desc(gate_diag);
            const uint64_t up_desc = make_tcgen05_tf32_diag_desc(up_diag);
            if (warp_id == 0 && lane == 0) {
              tcgen05_mma_tf32_cta1_ts(
                  tcgen_gate_scaled_tmem, partial_tmem, gate_desc, ts_idesc, enable_acc);
              tcgen05_mma_tf32_cta1_ts(
                  tcgen_up_scaled_tmem, partial_tmem, up_desc, ts_idesc, enable_acc);
            }
          }
          if (warp_id == 0 && lane == 0) {
            tcgen05_commit_group1(&tcgen_mma_barrier);
          }
          if (threadIdx.x == 0) {
            tcgen05_wait_mma_barrier_single(&tcgen_mma_barrier, tcgen_mma_phase_bit);
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          }
          __syncthreads();
          tcgen_mma_phase_bit ^= 1;
          tmem_scale_first = false;
        } else if (ablate_accum56) {
#pragma unroll
          for (int k_blk_local = 0; k_blk_local < kStep1TmaRawChunkBlocks; ++k_blk_local) {
            uint8_t* smem_A_for_mma =
                reinterpret_cast<uint8_t*>(A_cur) +
                static_cast<int64_t>(k_blk_local) * kStep1TcgenACompactBytes;
            issue_mma_64x8x128_tcgen05_no_wait(
                smem_A_for_mma, B_cur, k_blk_local, lane, warp_id,
                tcgen_tmem_base_ptr, !ablate_first_mma, effective_tcgen_accum_mode);
            ablate_first_mma = false;
          }
          if (warp_id == 0 && lane == 0) {
            tcgen05_commit_group1(&tcgen_mma_barrier);
          }
          if (threadIdx.x == 0) {
            tcgen05_wait_mma_barrier_single(&tcgen_mma_barrier, tcgen_mma_phase_bit);
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          }
          __syncthreads();
          tcgen_mma_phase_bit ^= 1;
        } else {
          const uint32_t tcgen_group_tmem_base = tcgen_tmem_base_ptr;
#pragma unroll
          for (int batch_base = 0; batch_base < kStep1TmaRawChunkBlocks;
               batch_base += kStep1TmaRawChunkBlocks) {
#pragma unroll
            for (int slot = 0; slot < kStep1TmaRawChunkBlocks; ++slot) {
              const int k_blk_local = batch_base + slot;
              uint8_t* smem_A_for_mma =
                  reinterpret_cast<uint8_t*>(A_cur) +
                  static_cast<int64_t>(k_blk_local) * kStep1TcgenACompactBytes;
              issue_mma_64x8x128_tcgen05_no_wait(
                  smem_A_for_mma,
                  B_cur,
                  k_blk_local,
                  lane,
                  warp_id,
                  tcgen_group_tmem_base + static_cast<uint32_t>(slot * kStep1TcgenN),
                  false,
                  effective_tcgen_accum_mode);
            }

            if (warp_id == 0 && lane == 0) {
              tcgen05_commit_group1(&tcgen_mma_barrier);
            }
            if ((k_group + 1 < kStep1TmaRawChunkGroups) && !prefetched_next_hidden) {
              load_hidden_wordtiles_group_sw128<kStep1CommThreads>(
                  reinterpret_cast<uint8_t*>(B_next), hidden_vec_dev, valid_token_idx,
                  packed_base, row_tile, n_rows,
                  (k_group + 1) * kStep1TmaRawChunkBlocks, kStep1TmaRawChunkBlocks);
              __syncthreads();
              prefetched_next_hidden = true;
            }
            if (threadIdx.x == 0) {
              tcgen05_wait_mma_barrier_single(
                  &tcgen_mma_barrier,
                  tcgen_mma_phase_bit);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
              asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            }
            __syncthreads();
            tcgen_mma_phase_bit ^= 1;

            if (warp_id < 4) {
              const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
              const int col_parity = (lane >> 1) & 1;
#pragma unroll
              for (int slot = 0; slot < kStep1TmaRawChunkBlocks; ++slot) {
                const int k_blk_local = batch_base + slot;
                const int k_blk = k_group * kStep1TmaRawChunkBlocks + k_blk_local;
                if (n_rows == 1) {
                  const float result0 = tcgen05_ld_16x64b_x1_accum_to_f32(
                      tcgen_group_tmem_base + static_cast<uint32_t>(slot * kStep1TcgenN),
                      effective_tcgen_accum_mode);
                  tcgen05_wait_ld_sync();
                  if (col_parity == 0) {
                    const int token_idx = valid_token_idx[packed_base + row_tile];
                    const float hidden_block_scale =
                        hidden_scale_dev[static_cast<int64_t>(k_blk) * t + token_idx];
                    if (result_row < kStep1OutRowsPerCta) {
                      const float gate_scale =
                          s13_all_dev[(expert * kStep1Gemm1OutBlocks + out_block128) *
                                          kStep1HiddenBlocks +
                                      k_blk];
                      gate_acc.v[0] += result0 * gate_scale * hidden_block_scale;
                    } else {
                      const float up_scale =
                          s13_all_dev[(expert * kStep1Gemm1OutBlocks +
                                       (out_block128 + kStep1IntermediateBlocks)) *
                                          kStep1HiddenBlocks +
                                      k_blk];
                      up_acc.v[0] += result0 * up_scale * hidden_block_scale;
                    }
                  }
                  continue;
                }
                float result_bits[4];
                tcgen05_ld_16x64b_x4_accum_to_f32(
                    result_bits,
                    tcgen_group_tmem_base + static_cast<uint32_t>(slot * kStep1TcgenN),
                    effective_tcgen_accum_mode,
                    col_parity);
                tcgen05_wait_ld_sync();
#pragma unroll
                for (int reg = 0; reg < 4; ++reg) {
                  const int rr = 2 * reg + col_parity;
                  if (rr >= n_rows) continue;
                  const int packed_idx = packed_base + row_tile + rr;
                  const int token_idx = valid_token_idx[packed_idx];
                  const float hidden_block_scale =
                      hidden_scale_dev[static_cast<int64_t>(k_blk) * t + token_idx];
                  if (result_row < kStep1OutRowsPerCta) {
                    const float gate_scale =
                        s13_all_dev[(expert * kStep1Gemm1OutBlocks + out_block128) *
                                        kStep1HiddenBlocks +
                                    k_blk];
                    float scale = gate_scale * hidden_block_scale;
                    if constexpr (!kFastTcgen) {
                      if (debug_output_mode == 3) scale = 1.0f;
                    }
                    gate_acc.v[rr] += result_bits[reg] * scale;
                  } else {
                    const float up_scale =
                        s13_all_dev[(expert * kStep1Gemm1OutBlocks +
                                     (out_block128 + kStep1IntermediateBlocks)) *
                                        kStep1HiddenBlocks +
                                    k_blk];
                    float scale = up_scale * hidden_block_scale;
                    if constexpr (!kFastTcgen) {
                      if (debug_output_mode == 4) scale = 1.0f;
                    }
                    up_acc.v[rr] += result_bits[reg] * scale;
                  }
                }
              }
            }
            __syncthreads();
          }
        }

        if (k_group + 1 < kStep1TmaRawChunkGroups) {
          if (!prefetched_next_hidden) {
            load_hidden_wordtiles_group_sw128<kStep1CommThreads>(
                reinterpret_cast<uint8_t*>(B_next), hidden_vec_dev, valid_token_idx,
                packed_base, row_tile, n_rows,
                (k_group + 1) * kStep1TmaRawChunkBlocks, kStep1TmaRawChunkBlocks);
            __syncthreads();
          }
          wait_weight_group_tma_combined(&weight_group_barrier[0]);
        }
      }
      if (use_tmem_scale_mma && warp_id < 4) {
        const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
        const int col_parity = (lane >> 1) & 1;
        float result_bits[4];
        const uint32_t read_tmem =
            result_row < kStep1OutRowsPerCta ? tcgen_gate_scaled_tmem : tcgen_up_scaled_tmem;
        tcgen05_ld_16x64b_x4(result_bits, read_tmem);
        tcgen05_wait_ld_sync();
#pragma unroll
        for (int reg = 0; reg < 4; ++reg) {
          const int rr = 2 * reg + col_parity;
          if (rr >= n_rows) continue;
          if (result_row < kStep1OutRowsPerCta) {
            gate_acc.v[rr] += result_bits[reg];
          } else {
            up_acc.v[rr] += result_bits[reg];
          }
        }
      }
#endif
    } else {
    for (int k_group = 0; k_group < kStep1TmaRawChunkGroups; ++k_group) {
      if (use_direct_tma_sw128) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        issue_weight_tma_sw128_64x1024(
            reinterpret_cast<uint8_t*>(smem_A_group), w13_tma_desc, expert,
            out_row_base, k_group, &weight_group_barrier[0]);
#else
        load_weight_group_vec_combined<kStep1CommThreads>(
            smem_A_group, w13_all_vec_dev, expert, out_row_base, k_group);
#endif
      } else if (use_tma) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        issue_weight_group_tma_combined(
            smem_A_group, w13_tma_desc, expert, out_row_base, kStep1OutRowsPerCta,
            kStep1TmaRawChunkBlocks, k_group, &weight_group_barrier[0]);
#else
        load_weight_group_vec_combined<kStep1CommThreads>(
            smem_A_group, w13_all_vec_dev, expert, out_row_base, k_group);
#endif
      } else {
        load_weight_group_vec_combined<kStep1CommThreads>(
            smem_A_group, w13_all_vec_dev, expert, out_row_base, k_group);
      }

      if (use_direct_b_sw128) {
        load_hidden_wordtiles_group_sw128<kStep1CommThreads>(
            reinterpret_cast<uint8_t*>(smem_B_group), hidden_vec_dev, valid_token_idx,
            packed_base, row_tile, n_rows,
            k_group * kStep1TmaRawChunkBlocks, kStep1TmaRawChunkBlocks);
      } else {
        load_hidden_wordtiles_group<kStep1CommThreads>(
            smem_B_group, hidden_words_dev, valid_token_idx, packed_base, row_tile, n_rows,
            k_group * kStep1TmaRawChunkBlocks, kStep1TmaRawChunkBlocks);
      }

      __syncthreads();

      if (use_tma) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        wait_weight_group_tma_combined(&weight_group_barrier[0]);
#endif
      }

      if (ablate_accum56) {
#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#pragma unroll
        for (int k_blk_local = 0; k_blk_local < kStep1TmaRawChunkBlocks; ++k_blk_local) {
          uint8_t* smem_A_for_mma =
              reinterpret_cast<uint8_t*>(smem_A_group) +
              static_cast<int64_t>(k_blk_local) * kStep1TcgenACompactBytes;
          issue_mma_64x8x128_tcgen05_no_wait(
              smem_A_for_mma, smem_B_group, k_blk_local, lane, warp_id,
              tcgen_tmem_base_ptr, !ablate_first_mma, effective_tcgen_accum_mode);
          ablate_first_mma = false;
        }
        if (warp_id == 0 && lane == 0) {
          tcgen05_commit_group1(&tcgen_mma_barrier);
        }
        if (threadIdx.x == 0) {
          tcgen05_wait_mma_barrier_single(&tcgen_mma_barrier, tcgen_mma_phase_bit);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        }
        __syncthreads();
        tcgen_mma_phase_bit ^= 1;
#endif
        continue;
      }

#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
      if (use_direct_tma_sw128 && use_direct_b_sw128) {
#pragma unroll
        for (int batch_base = 0; batch_base < kStep1TmaRawChunkBlocks; batch_base += 4) {
#pragma unroll
          for (int slot = 0; slot < 4; ++slot) {
            const int k_blk_local = batch_base + slot;
            uint8_t* smem_A_for_mma =
                reinterpret_cast<uint8_t*>(smem_A_group) +
                static_cast<int64_t>(k_blk_local) * kStep1TcgenACompactBytes;
            issue_mma_64x8x128_tcgen05_no_wait(
                smem_A_for_mma,
                smem_B_group,
                k_blk_local,
                lane,
                warp_id,
                tcgen_tmem_base_ptr + static_cast<uint32_t>(slot * kStep1TcgenN),
                false,
                effective_tcgen_accum_mode);
          }

          if (warp_id == 0 && lane == 0) {
            tcgen05_commit_group1(&tcgen_mma_barrier);
          }
          if (threadIdx.x == 0) {
            tcgen05_wait_mma_barrier_single(&tcgen_mma_barrier, tcgen_mma_phase_bit);
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          }
          __syncthreads();
          tcgen_mma_phase_bit ^= 1;

          if (warp_id < 4) {
            const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
            const int col_parity = (lane >> 1) & 1;
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
              const int k_blk_local = batch_base + slot;
              const int k_blk = k_group * kStep1TmaRawChunkBlocks + k_blk_local;
              float result_bits[4];
              tcgen05_ld_16x64b_x4_accum_to_f32(
                  result_bits,
                  tcgen_tmem_base_ptr + static_cast<uint32_t>(slot * kStep1TcgenN),
                  effective_tcgen_accum_mode,
                  col_parity);
              tcgen05_wait_ld_sync();
#pragma unroll
              for (int reg = 0; reg < 4; ++reg) {
                const int rr = 2 * reg + col_parity;
                if (rr >= n_rows) continue;
                const int packed_idx = packed_base + row_tile + rr;
                const int token_idx = valid_token_idx[packed_idx];
                const float hidden_block_scale =
                    hidden_scale_dev[static_cast<int64_t>(k_blk) * t + token_idx];
                if (result_row < kStep1OutRowsPerCta) {
                  const float gate_scale =
                      s13_all_dev[(expert * kStep1Gemm1OutBlocks + out_block128) *
                                      kStep1HiddenBlocks +
                                  k_blk];
                  float scale = gate_scale * hidden_block_scale;
                  if constexpr (!kFastTcgen) {
                    if (debug_output_mode == 3) scale = 1.0f;
                  }
                  gate_acc.v[rr] += result_bits[reg] * scale;
                } else {
                  const float up_scale =
                      s13_all_dev[(expert * kStep1Gemm1OutBlocks +
                                   (out_block128 + kStep1IntermediateBlocks)) *
                                      kStep1HiddenBlocks +
                                  k_blk];
                  float scale = up_scale * hidden_block_scale;
                  if constexpr (!kFastTcgen) {
                    if (debug_output_mode == 4) scale = 1.0f;
                  }
                  up_acc.v[rr] += result_bits[reg] * scale;
                }
              }
            }
          }
          __syncthreads();
        }
        continue;
      }
#endif

#pragma unroll
      for (int k_blk_local = 0; k_blk_local < kStep1TmaRawChunkBlocks; ++k_blk_local) {
        const int k_blk = k_group * kStep1TmaRawChunkBlocks + k_blk_local;
        const float gate_scale =
            s13_all_dev[(expert * kStep1Gemm1OutBlocks + out_block128) * kStep1HiddenBlocks + k_blk];
	        const float up_scale =
	            s13_all_dev[(expert * kStep1Gemm1OutBlocks +
	                         (out_block128 + kStep1IntermediateBlocks)) * kStep1HiddenBlocks + k_blk];
        uint8_t* smem_A_for_mma =
            use_direct_tma_sw128
                ? reinterpret_cast<uint8_t*>(smem_A_group) +
                      static_cast<int64_t>(k_blk_local) * kStep1TcgenACompactBytes
                : smem_A_tcgen;
	#pragma unroll 1
        for (int once = 0; once < 1; ++once) {
          (void)once;
          mma_64x8x128_tcgen05(
              gate_acc,
              up_acc,
              smem_A_for_mma,
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
              lane,
              warp_id,
              debug_output_mode,
#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
              tcgen_tmem_base_ptr,
              &tcgen_mma_barrier,
              tcgen_mma_phase_bit,
              use_direct_tma_sw128,
              use_direct_b_sw128,
              effective_tcgen_accum_mode
#else
              0u,
              nullptr,
              tcgen_mma_phase_bit,
              false,
              use_direct_b_sw128,
              effective_tcgen_accum_mode
#endif
              );
        }
      }

      __syncthreads();
    }
    }

    if (ablate_accum56 && warp_id < 4) {
#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
      float result_bits[4];
      const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
      const int col_parity = (lane >> 1) & 1;
      tcgen05_ld_16x64b_x4_accum_to_f32(
          result_bits, tcgen_tmem_base_ptr, effective_tcgen_accum_mode, col_parity);
      tcgen05_wait_ld_sync();
#pragma unroll
      for (int reg = 0; reg < 4; ++reg) {
        const int rr = 2 * reg + col_parity;
        if (rr >= n_rows) continue;
        if (result_row < kStep1OutRowsPerCta) {
          gate_acc.v[rr] = result_bits[reg];
        } else {
          up_acc.v[rr] = result_bits[reg];
        }
      }
#endif
    }

    if (warp_id >= 2 && warp_id < 4) {
      const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
      const int col_parity = (lane >> 1) & 1;
#pragma unroll
      for (int reg = 0; reg < 4; ++reg) {
        const int rr = 2 * reg + col_parity;
        if (rr >= n_rows) continue;
        const int exchange_row = result_row - kStep1OutRowsPerCta;
        tcgen_up_acc_smem[rr * kStep1OutRowsPerCta + exchange_row] = up_acc.v[rr];
      }
    }
    __syncthreads();

    const bool do_store = (warp_id < 2);

    if (do_store) {
      const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
      const int col_parity = (lane >> 1) & 1;
      const int out_col = out_row_base + result_row;
#pragma unroll
      for (int reg = 0; reg < 4; ++reg) {
        const int rr = 2 * reg + col_parity;
        if (rr >= n_rows) continue;
        const int packed_row = packed_base + row_tile + rr;
        const float gate_v = gate_acc.v[rr];
        const float up_v = tcgen_up_acc_smem[rr * kStep1OutRowsPerCta + result_row];
        float value = gate_v * silu_device(up_v);
        if constexpr (!kFastTcgen) {
          if (debug_output_mode == 1 || debug_output_mode == 3 || debug_output_mode == 98) {
            value = gate_v;
          } else if (debug_output_mode == 2 || debug_output_mode == 4) {
            value = up_v;
          }
        }
        c_perm_all_dev[static_cast<int64_t>(packed_row) * kStep1Intermediate + out_col] = value;
      }
    }
    __syncthreads();
  }

#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (warp_id == 0) {
    tcgen05_dealloc_cta1_cols_64(tcgen_tmem_base_ptr);
  }
  __syncthreads();
  if (warp_id == 0) {
    tcgen05_relinquish_alloc_permit_cta1();
  }
  __syncthreads();
#endif
}

inline cudaError_t EnsureStep1AllExpertsDirectAttributes() {
  static bool attrs_set = false;
  if (attrs_set) return cudaSuccess;
  const std::size_t smem_bytes = step1_smem_bytes();
  cudaError_t st = cudaFuncSetAttribute(
      step1_gemm1_swiglu_direct_kernel<-1>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  if (st != cudaSuccess) {
    return st;
  }
  st = cudaFuncSetAttribute(
      step1_gemm1_swiglu_direct_kernel<2>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  if (st != cudaSuccess) {
    return st;
  }
  attrs_set = true;
  return cudaSuccess;
}

inline cudaError_t RunStep1AllExpertsDirect(
    const uint8_t* hidden_fp8_dev,
    const float* hidden_scale_dev,
    int64_t t,
    const int* expert_counts_dev,
    const int* expert_offsets_dev,
    const int* permuted_token_ids_dev,
    const int* active_experts_dev,
    int active_expert_count,
    const uint8_t* w13_all_dev,
    const float* s13_all_dev,
    const void* w13_tma_desc,
    int debug_output_mode,
    float* c_perm_all_dev,
    cudaStream_t stream,
    bool direct_tma_sw128 = false,
    bool direct_b_sw128 = false,
    int tcgen_accum_mode = 0,
    bool check_launch_error = true) {
  const std::size_t smem_bytes = step1_smem_bytes();
  const bool use_fast_tcgen =
      (w13_tma_desc != nullptr) && direct_tma_sw128 && direct_b_sw128 &&
      debug_output_mode == 0 && tcgen_accum_mode == 2;

  cudaError_t attr_status = EnsureStep1AllExpertsDirectAttributes();
  if (attr_status != cudaSuccess) return attr_status;
  const int grid_experts =
      active_expert_count > 0 ? active_expert_count : kStep1LocalExperts;
  dim3 grid(kStep1OutTilesPerExpert, grid_experts, 1);
  dim3 block(kStep1CommThreads, 1, 1);
  if (use_fast_tcgen) {
    step1_gemm1_swiglu_direct_kernel<2><<<grid, block, smem_bytes, stream>>>(
        hidden_fp8_dev,
        hidden_scale_dev,
        t,
        expert_counts_dev,
        expert_offsets_dev,
        permuted_token_ids_dev,
        active_experts_dev,
        w13_all_dev,
        s13_all_dev,
        nullptr,
        w13_tma_desc,
        true,
        true,
        0,
        2,
        c_perm_all_dev);
  } else {
    step1_gemm1_swiglu_direct_kernel<-1><<<grid, block, smem_bytes, stream>>>(
        hidden_fp8_dev,
        hidden_scale_dev,
        t,
        expert_counts_dev,
        expert_offsets_dev,
        permuted_token_ids_dev,
        active_experts_dev,
        w13_all_dev,
        s13_all_dev,
        nullptr,
        w13_tma_desc,
        direct_tma_sw128,
        direct_b_sw128,
        debug_output_mode,
        tcgen_accum_mode,
        c_perm_all_dev);
  }

  return check_launch_error ? cudaGetLastError() : cudaSuccess;
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
		  const uint32_t smem_raw_addr = cute::cast_smem_ptr_to_uint(smem_raw);
		  const std::size_t smem_align_pad = (1024u - (smem_raw_addr & 1023u)) & 1023u;
			  const int comm_gate_up_tiles = (comm_tma_mode == 3) ? 1 : 2;
			  const std::size_t weight_group_bytes =
			      static_cast<std::size_t>(comm_gate_up_tiles) * comm_out_rows *
	          comm_h_inner_bytes * comm_h_tiles;
		  uint32_t* smem_A_group = reinterpret_cast<uint32_t*>(smem_raw + smem_align_pad);
		  uint32_t* smem_B_words =
		      reinterpret_cast<uint32_t*>(smem_raw + smem_align_pad + weight_group_bytes);
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
		              comm_h_tiles, k_group, &weight_group_barrier, comm_gate_up_tiles);
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

  static std::size_t cached_smem_bytes = 0;
  static std::size_t cached_tma_smem_bytes = 0;
  if (cached_smem_bytes != smem_bytes) {
    cudaError_t st = cudaFuncSetAttribute(
        step1_comm_only_direct_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    if (st != cudaSuccess) {
      return st;
    }
    cached_smem_bytes = smem_bytes;
  }
  if (cached_tma_smem_bytes != tma_smem_bytes) {
    cudaError_t st = cudaFuncSetAttribute(
        step1_comm_only_direct_kernel<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(tma_smem_bytes));
    if (st != cudaSuccess) {
      return st;
    }
    cached_tma_smem_bytes = tma_smem_bytes;
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
