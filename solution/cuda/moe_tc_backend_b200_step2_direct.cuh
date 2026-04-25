#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda/ptx>
#include <cmath>
#include <cstdint>

namespace direct_backend {

static constexpr int kStep2Hidden = 7168;
static constexpr int kStep2Intermediate = 2048;
static constexpr int kStep2Block = 128;
static constexpr int kStep2TcgenM = 64;
static constexpr int kStep2TcgenN = 8;
static constexpr int kStep2Threads = 128;
static constexpr int kStep2LocalExperts = 32;
static constexpr int kStep2IntermediateBlocks = kStep2Intermediate / kStep2Block;
static constexpr int kStep2TmaKGroupBlocks = 8;
static constexpr int kStep2TmaKGroups = kStep2IntermediateBlocks / kStep2TmaKGroupBlocks;
static constexpr int kStep2HiddenTiles64 = kStep2Hidden / kStep2TcgenM;
static constexpr int kStep2HiddenScaleBlocks = kStep2Hidden / kStep2Block;
static constexpr int kStep2TcgenACompactBytes = kStep2TcgenM * kStep2Block;
static constexpr int kStep2TcgenBCompactBytes = kStep2TcgenN * kStep2Block;
static constexpr int kStep2W2TmaGroupBytes =
    kStep2TcgenM * kStep2Block * kStep2TmaKGroupBlocks;
static constexpr uint32_t kStep2TcgenLayoutSwizzle128B = 2;
static constexpr int kStep2K32IssuesPerBlock = 4;
static constexpr float kStep2CScaleDenom = 448.0f;

enum Step2PipelineVariant : int {
  kStep2PipeM64N8K8DbBatch2 = 0,
  kStep2PipeM64N8K4DbBatch2 = 1,
  kStep2PipeM64N16K8DbBatch2 = 2,
  kStep2PipeM64N16K4DbBatch2 = 3,
  kStep2PipeM128N8K8DbBatch2 = 4,
};

static_assert(kStep2Hidden % kStep2TcgenM == 0, "Step2 H must divide the tcgen05 M tile.");
static_assert(kStep2Intermediate % kStep2Block == 0, "Step2 I must divide K128.");
static_assert(kStep2IntermediateBlocks % kStep2TmaKGroupBlocks == 0,
              "Step2 W2 TMA grouping requires exact K128 division.");

__host__ __device__ __forceinline__ int step2_min_int(int a, int b) {
  return a < b ? a : b;
}

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

__device__ __forceinline__ uint8_t float_to_fp8_e4m3fn_device_step2(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float step2_subwarp8_reduce_max(float v, uint32_t mask) {
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(mask, v, offset, 8));
  }
  return v;
}

__device__ __forceinline__ uint32_t SmemPtrStep2(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ bool issue_step2_w2_tma_sw128_64x1024(
    uint8_t* smem_w2_tma,
    const void* __restrict__ w2_tma_desc,
    int expert,
    int h_tile64,
    int k_group,
    uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (w2_tma_desc == nullptr) return false;
  if (threadIdx.x == 0) {
    const int32_t coords[4] = {
        0,
        h_tile64 * kStep2TcgenM,
        k_group * kStep2TmaKGroupBlocks,
        expert};
    cuda::ptx::mbarrier_init(barrier, 1);
    cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global, cuda::ptx::cta_group_1,
        smem_w2_tma, w2_tma_desc, coords, barrier);
    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared,
        barrier, static_cast<uint32_t>(kStep2W2TmaGroupBytes));
  }
  return true;
#else
  (void)smem_w2_tma;
  (void)w2_tma_desc;
  (void)expert;
  (void)h_tile64;
  (void)k_group;
  (void)barrier;
  return false;
#endif
}

template <int KGroupBlocks>
__device__ __forceinline__ bool issue_step2_w2_tma_sw128_64xkgroup(
    uint8_t* smem_w2_tma,
    const void* __restrict__ w2_tma_desc,
    int expert,
    int h_tile64,
    int k_group,
    uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (w2_tma_desc == nullptr) return false;
  if (threadIdx.x == 0) {
    const int32_t coords[4] = {
        0,
        h_tile64 * kStep2TcgenM,
        k_group * KGroupBlocks,
        expert};
    cuda::ptx::mbarrier_init(barrier, 1);
    cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global, cuda::ptx::cta_group_1,
        smem_w2_tma, w2_tma_desc, coords, barrier);
    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared,
        barrier, static_cast<uint32_t>(kStep2TcgenM * kStep2Block * KGroupBlocks));
  }
  return true;
#else
  (void)smem_w2_tma;
  (void)w2_tma_desc;
  (void)expert;
  (void)h_tile64;
  (void)k_group;
  (void)barrier;
  return false;
#endif
}

__device__ __forceinline__ void wait_step2_w2_tma(uint64_t* barrier) {
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

// Step2 direct path for a single expert slice (legacy path kept for call-site compatibility).
static __global__ void step2_gemm2_scatter_direct_kernel(const float* __restrict__ c_perm_dev, int n_rows,
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
static __global__ void step2_gemm2_scatter_all_experts_direct_kernel(
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

  const int t_valid = expert_t_valid[expert];
  if (t_valid <= 0) return;
  const int row_start = expert_offset[expert];

  const int intermediate_blocks = kStep2Intermediate / kStep2Block;
  const size_t w2_expert_elems = static_cast<size_t>(kStep2Hidden) * kStep2Intermediate;
  const size_t s2_expert_elems = static_cast<size_t>(kStep2Hidden / kStep2Block) * intermediate_blocks;
  const uint8_t* w2_e = w2_all_dev + static_cast<size_t>(expert) * w2_expert_elems;
  const float* s2_e = s2_all_dev + static_cast<size_t>(expert) * s2_expert_elems;

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
      for (int u = 0; u < kStep2Block; ++u) {
        const int i = i0 + u;
        const float cv = c_row[i];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          const int h = hb * kStep2Block + lane + v * 32;
          const float wv = fp8_e4m3fn_to_float_device_step2(
              w2_e[static_cast<int64_t>(h) * kStep2Intermediate + i]);
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

__device__ __forceinline__ constexpr uint32_t step2_make_idesc_f8f6f4_f32_dense(int m, int n) {
  return (1u << 4) |
         (static_cast<uint32_t>(n >> 3) << 17) |
         (static_cast<uint32_t>(m >> 4) << 24);
}

__device__ __forceinline__ constexpr uint32_t step2_make_idesc_tf32_ts_f32_dense(int m, int n) {
  return (1u << 4) | (2u << 7) | (2u << 10) |
         (static_cast<uint32_t>(n >> 3) << 17) |
         (static_cast<uint32_t>(m >> 4) << 24);
}

__device__ __forceinline__ uint64_t step2_make_core_matrix_desc_group8_k128(const void* smem) {
  const uint32_t matrix_start_aligned = SmemPtrStep2(smem) & ~0xFu;
  const uint32_t lbo = 16u;
  const uint32_t sbo = 8u * static_cast<uint32_t>(kStep2Block);
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
  desc |= static_cast<uint64_t>(kStep2TcgenLayoutSwizzle128B & 0x7u) << 61;
  return desc;
}

__device__ __forceinline__ uint64_t step2_make_tf32_diag_desc(const void* smem) {
  const uint32_t matrix_start_aligned = SmemPtrStep2(smem) & ~0xFu;
  const uint32_t lbo = kStep2TcgenN * 16u;
  const uint32_t sbo = kStep2TcgenN * 16u;
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>(matrix_start_aligned >> 4);
  desc |= static_cast<uint64_t>((lbo & 0x3ffffu) >> 4) << 16;
  desc |= static_cast<uint64_t>((sbo & 0x3ffffu) >> 4) << 32;
  desc |= static_cast<uint64_t>(1u) << 46;
  desc |= static_cast<uint64_t>(0xb0u) << 53;
  return desc;
}

__device__ __forceinline__ float step2_tf32_rne_prebias(float value) {
  return __uint_as_float(__float_as_uint(value) + 0x1000u);
}

__device__ __forceinline__ void step2_tcgen05_mbarrier_init(uint64_t* barrier, uint32_t count) {
  const uint32_t bar_ptr = SmemPtrStep2(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :: "r"(bar_ptr), "r"(count)
               : "memory");
}

__device__ __forceinline__ void step2_tcgen05_commit_group1(const uint64_t* barrier) {
  const uint32_t bar_ptr = SmemPtrStep2(barrier);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(bar_ptr)
               : "memory");
}

__device__ __forceinline__ void step2_tcgen05_wait_mma_barrier_single(uint64_t* barrier, int phase) {
  const uint32_t bar_ptr = SmemPtrStep2(barrier);
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

__device__ __forceinline__ uint32_t step2_tcgen05_alloc_cta1_cols_64(uint32_t* smem_out_taddr) {
  const uint32_t smem_addr = SmemPtrStep2(smem_out_taddr);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"
               :: "r"(smem_addr)
               : "memory");
  __syncwarp();
  uint32_t taddr;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr) : "r"(smem_addr) : "memory");
  return taddr;
}

__device__ __forceinline__ void step2_tcgen05_dealloc_cta1_cols_64(uint32_t taddr) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;"
               :: "r"(taddr)
               : "memory");
}

__device__ __forceinline__ uint32_t step2_tcgen05_alloc_cta1_cols_128(uint32_t* smem_out_taddr) {
  const uint32_t smem_addr = SmemPtrStep2(smem_out_taddr);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"
               :: "r"(smem_addr)
               : "memory");
  __syncwarp();
  uint32_t taddr;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr) : "r"(smem_addr) : "memory");
  return taddr;
}

__device__ __forceinline__ void step2_tcgen05_dealloc_cta1_cols_128(uint32_t taddr) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;"
               :: "r"(taddr)
               : "memory");
}

__device__ __forceinline__ void step2_tcgen05_relinquish_alloc_permit_cta1() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void step2_tcgen05_wait_ld_sync() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void step2_tcgen05_ld_16x64b_x4(float (&dst)[4], uint32_t taddr) {
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

__device__ __forceinline__ void step2_tcgen05_mma_f8f6f4_cta1_ss(
    uint32_t d_tmem,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t idesc,
    bool enable_input_d) {
  const uint32_t enable_u32 = enable_input_d ? 1u : 0u;
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

__device__ __forceinline__ void step2_tcgen05_mma_tf32_cta1_ts(
    uint32_t d_tmem,
    uint32_t a_tmem,
    uint64_t b_desc,
    uint32_t idesc,
    bool enable_input_d) {
  const uint32_t enable_u32 = enable_input_d ? 1u : 0u;
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

__device__ __forceinline__ void stage_step2_w2_a_fp8_64x128(
    uint8_t* __restrict__ smem_A_tcgen,
    const uint8_t* __restrict__ w2_all_dev,
    int expert,
    int h_tile64,
    int k_blk) {
  constexpr int kVecBytes = 16;
  constexpr int kVecsPerRow = kStep2Block / kVecBytes;
  constexpr int kStageVecs = kStep2TcgenM * kVecsPerRow;
  const int h_base = h_tile64 * kStep2TcgenM;
#pragma unroll
  for (int iter = 0; iter < (kStageVecs + kStep2Threads - 1) / kStep2Threads; ++iter) {
    const int linear_vec = threadIdx.x + iter * kStep2Threads;
    if (linear_vec >= kStageVecs) continue;
    const int m = linear_vec / kVecsPerRow;
    const int chunk16 = linear_vec - m * kVecsPerRow;
    const int h = h_base + m;
    const int i = k_blk * kStep2Block + chunk16 * kVecBytes;
    const int row_group = m >> 3;
    const int row_in8 = m & 7;
    const int phys_chunk16 = chunk16 ^ row_in8;
    const int64_t w_idx =
        (static_cast<int64_t>(expert) * kStep2Hidden + h) * kStep2Intermediate + i;
    const int dst = row_group * 8 * kStep2Block + row_in8 * kStep2Block +
                    phys_chunk16 * kVecBytes;
    *reinterpret_cast<uint4*>(smem_A_tcgen + dst) =
        *reinterpret_cast<const uint4*>(w2_all_dev + w_idx);
  }
}

__device__ __forceinline__ void load_step2_w2_group_direct_rowmajor(
    uint8_t* __restrict__ smem_w2_tma,
    const uint8_t* __restrict__ w2_all_dev,
    int expert,
    int h_tile64,
    int k_group) {
  constexpr int kVecBytes = 16;
  constexpr int kVecsPerK128 = kStep2Block / kVecBytes;
  constexpr int kVecsPerRow = kStep2TmaKGroupBlocks * kVecsPerK128;
  constexpr int kTotalVecs = kStep2TcgenM * kVecsPerRow;
  const int h_base = h_tile64 * kStep2TcgenM;
#pragma unroll
  for (int iter = 0; iter < (kTotalVecs + kStep2Threads - 1) / kStep2Threads; ++iter) {
    const int linear_vec = threadIdx.x + iter * kStep2Threads;
    if (linear_vec >= kTotalVecs) continue;
    const int m = linear_vec / kVecsPerRow;
    const int row_vec = linear_vec - m * kVecsPerRow;
    const int k_sub = row_vec / kVecsPerK128;
    const int vec_in_k128 = row_vec - k_sub * kVecsPerK128;
    const int h = h_base + m;
    const int i =
        (k_group * kStep2TmaKGroupBlocks + k_sub) * kStep2Block + vec_in_k128 * kVecBytes;
    const int64_t w_idx =
        (static_cast<int64_t>(expert) * kStep2Hidden + h) * kStep2Intermediate + i;
    const int dst = (m * kStep2TmaKGroupBlocks + k_sub) * kStep2Block +
                    vec_in_k128 * kVecBytes;
    *reinterpret_cast<uint4*>(smem_w2_tma + dst) =
        *reinterpret_cast<const uint4*>(w2_all_dev + w_idx);
  }
}

template <int KGroupBlocks>
__device__ __forceinline__ void load_step2_w2_group_direct_rowmajor_kgroup(
    uint8_t* __restrict__ smem_w2_tma,
    const uint8_t* __restrict__ w2_all_dev,
    int expert,
    int h_tile64,
    int k_group) {
  constexpr int kVecBytes = 16;
  constexpr int kVecsPerK128 = kStep2Block / kVecBytes;
  constexpr int kVecsPerRow = KGroupBlocks * kVecsPerK128;
  constexpr int kTotalVecs = kStep2TcgenM * kVecsPerRow;
  const int h_base = h_tile64 * kStep2TcgenM;
#pragma unroll
  for (int iter = 0; iter < (kTotalVecs + kStep2Threads - 1) / kStep2Threads; ++iter) {
    const int linear_vec = threadIdx.x + iter * kStep2Threads;
    if (linear_vec >= kTotalVecs) continue;
    const int m = linear_vec / kVecsPerRow;
    const int row_vec = linear_vec - m * kVecsPerRow;
    const int k_sub = row_vec / kVecsPerK128;
    const int vec_in_k128 = row_vec - k_sub * kVecsPerK128;
    const int h = h_base + m;
    const int i = (k_group * KGroupBlocks + k_sub) * kStep2Block +
                  vec_in_k128 * kVecBytes;
    const int64_t w_idx =
        (static_cast<int64_t>(expert) * kStep2Hidden + h) * kStep2Intermediate + i;
    const int dst = (m * KGroupBlocks + k_sub) * kStep2Block + vec_in_k128 * kVecBytes;
    *reinterpret_cast<uint4*>(smem_w2_tma + dst) =
        *reinterpret_cast<const uint4*>(w2_all_dev + w_idx);
  }
}

__device__ __forceinline__ void repack_step2_w2_tma_group_to_a_fp8_64x128(
    uint8_t* __restrict__ smem_A_tcgen,
    const uint8_t* __restrict__ smem_w2_tma,
    int k_sub) {
  constexpr int kVecBytes = 16;
  constexpr int kVecsPerRow = kStep2Block / kVecBytes;
  constexpr int kStageVecs = kStep2TcgenM * kVecsPerRow;
#pragma unroll
  for (int iter = 0; iter < (kStageVecs + kStep2Threads - 1) / kStep2Threads; ++iter) {
    const int linear_vec = threadIdx.x + iter * kStep2Threads;
    if (linear_vec >= kStageVecs) continue;
    const int m = linear_vec / kVecsPerRow;
    const int chunk16 = linear_vec - m * kVecsPerRow;
    const int row_group = m >> 3;
    const int row_in8 = m & 7;
    const int phys_chunk16 = chunk16 ^ row_in8;
    const int src = (m * kStep2TmaKGroupBlocks + k_sub) * kStep2Block +
                    chunk16 * kVecBytes;
    const int dst = row_group * 8 * kStep2Block + row_in8 * kStep2Block +
                    phys_chunk16 * kVecBytes;
    *reinterpret_cast<uint4*>(smem_A_tcgen + dst) =
        *reinterpret_cast<const uint4*>(smem_w2_tma + src);
  }
}

template <int KGroupBlocks>
__device__ __forceinline__ void repack_step2_w2_tma_group_to_a_fp8_64x128_kgroup(
    uint8_t* __restrict__ smem_A_tcgen,
    const uint8_t* __restrict__ smem_w2_tma,
    int k_sub) {
  constexpr int kVecBytes = 16;
  constexpr int kVecsPerRow = kStep2Block / kVecBytes;
  constexpr int kStageVecs = kStep2TcgenM * kVecsPerRow;
#pragma unroll
  for (int iter = 0; iter < (kStageVecs + kStep2Threads - 1) / kStep2Threads; ++iter) {
    const int linear_vec = threadIdx.x + iter * kStep2Threads;
    if (linear_vec >= kStageVecs) continue;
    const int m = linear_vec / kVecsPerRow;
    const int chunk16 = linear_vec - m * kVecsPerRow;
    const int row_group = m >> 3;
    const int row_in8 = m & 7;
    const int phys_chunk16 = chunk16 ^ row_in8;
    const int src = (m * KGroupBlocks + k_sub) * kStep2Block + chunk16 * kVecBytes;
    const int dst = row_group * 8 * kStep2Block + row_in8 * kStep2Block +
                    phys_chunk16 * kVecBytes;
    *reinterpret_cast<uint4*>(smem_A_tcgen + dst) =
        *reinterpret_cast<const uint4*>(smem_w2_tma + src);
  }
}

__device__ __forceinline__ void zero_step2_tf32_diag_scale(float* __restrict__ smem_diag) {
  constexpr int kDiagElems = 2 * kStep2TcgenN * 4;
  constexpr int kTotalElems = kStep2K32IssuesPerBlock * kDiagElems;
#pragma unroll
  for (int iter = 0; iter < (kTotalElems + kStep2Threads - 1) / kStep2Threads; ++iter) {
    const int linear = threadIdx.x + iter * kStep2Threads;
    if (linear >= kTotalElems) continue;
    smem_diag[linear] = 0.0f;
  }
}

static __global__ void step2_prequant_cperm_kernel(
    const float* __restrict__ c_perm_all_dev,
    int total_routed,
    uint8_t* __restrict__ c_perm_q_dev,
    float* __restrict__ c_perm_scale_dev) {
  const int k_blk = blockIdx.x;
  const int slot = blockIdx.y;
  if (slot >= total_routed) return;
  const int lane = threadIdx.x & 31;
  if (threadIdx.x >= 32) return;

  constexpr int kFloatsPerVec = 4;
  const float* src = c_perm_all_dev + static_cast<int64_t>(slot) * kStep2Intermediate +
                     k_blk * kStep2Block + lane * kFloatsPerVec;
  float4 v4 = *reinterpret_cast<const float4*>(src);
  float local_max = fmaxf(fmaxf(fabsf(v4.x), fabsf(v4.y)),
                          fmaxf(fabsf(v4.z), fabsf(v4.w)));
  const int issue = lane >> 3;
  const int sublane = lane & 7;
  const uint32_t issue_mask = 0xffu << (issue * 8);
  const float reduced_max_abs = step2_subwarp8_reduce_max(local_max, issue_mask);
  const float max_abs = __shfl_sync(issue_mask, reduced_max_abs, issue * 8);
  const float c_scale =
      max_abs > 0.0f ? (max_abs / kStep2CScaleDenom) : 1.0f;
  if (sublane == 0) {
    c_perm_scale_dev[static_cast<int64_t>(slot) * kStep2IntermediateBlocks *
                         kStep2K32IssuesPerBlock +
                     k_blk * kStep2K32IssuesPerBlock + issue] = c_scale;
  }
  const float inv_scale = 1.0f / c_scale;
  uint8_t* dst = c_perm_q_dev + static_cast<int64_t>(slot) * kStep2Intermediate +
                 k_blk * kStep2Block + lane * kFloatsPerVec;
  dst[0] = float_to_fp8_e4m3fn_device_step2(v4.x * inv_scale);
  dst[1] = float_to_fp8_e4m3fn_device_step2(v4.y * inv_scale);
  dst[2] = float_to_fp8_e4m3fn_device_step2(v4.z * inv_scale);
  dst[3] = float_to_fp8_e4m3fn_device_step2(v4.w * inv_scale);
}

static inline cudaError_t LaunchStep2PrequantCperm(
    const float* c_perm_all_dev,
    int total_routed,
    uint8_t* c_perm_q_dev,
    float* c_perm_scale_dev,
    cudaStream_t stream) {
  if (total_routed <= 0) return cudaSuccess;
  dim3 grid(kStep2IntermediateBlocks, total_routed);
  dim3 threads(32);
  step2_prequant_cperm_kernel<<<grid, threads, 0, stream>>>(
      c_perm_all_dev, total_routed, c_perm_q_dev, c_perm_scale_dev);
  return cudaGetLastError();
}

// Stage the prequantized Step1 output as tcgen B and update only the non-zero
// diagonal entries for the per-issue W2*C scale matrix.
__device__ __forceinline__ void stage_step2_cperm_q_b_fp8_8x128_update_diag_only(
    uint8_t* __restrict__ smem_B_tcgen,
    float* __restrict__ smem_diag,
    const uint8_t* __restrict__ c_perm_q_dev,
    const float* __restrict__ c_perm_scale_dev,
    const int* __restrict__ expert_offset,
    int expert,
    int row_tile,
    int n_rows,
    int k_blk,
    float w_scale) {
  constexpr int kBytesPerLane = 4;
  constexpr int kDiagElems = 2 * kStep2TcgenN * 4;
  const int row_start = expert_offset[expert];
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int issue = lane >> 3;
#pragma unroll
  for (int token_in_warp = 0; token_in_warp < 2; ++token_in_warp) {
    const int n = warp_id * 2 + token_in_warp;
    if (n >= n_rows) continue;
    const int slot = row_start + row_tile + n;
    const float c_scale =
        c_perm_scale_dev[static_cast<int64_t>(slot) * kStep2IntermediateBlocks *
                             kStep2K32IssuesPerBlock +
                         k_blk * kStep2K32IssuesPerBlock + issue];
    if ((lane & 7) == 0) {
      const int diag_slice = n >> 2;
      const int diag_k = n & 3;
      smem_diag[issue * kDiagElems + diag_slice * kStep2TcgenN * 4 + n * 4 + diag_k] =
          step2_tf32_rne_prebias(w_scale * c_scale);
    }
    const int k_base = lane * kBytesPerLane;
    const int chunk16 = k_base >> 4;
    const int byte_base = k_base & 15;
    const int phys_chunk16 = chunk16 ^ n;
    const uint8_t* src = c_perm_q_dev + static_cast<int64_t>(slot) * kStep2Intermediate +
                         k_blk * kStep2Block + k_base;
    uint8_t* dst = smem_B_tcgen + n * kStep2Block + phys_chunk16 * 16 + byte_base;
    *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(src);
  }
}

// Issue the FP8xFP8 partial GEMM for one K32 issue into temporary TMEM columns.
__device__ __forceinline__ void step2_issue_mma_64x8x128_f8f6f4_ss(
    const uint8_t* __restrict__ smem_A_tcgen,
    const uint8_t* __restrict__ smem_B_tcgen,
    uint32_t tmem_base,
    bool enable_input_d,
    int issue) {
  constexpr uint32_t kIdesc =
      step2_make_idesc_f8f6f4_f32_dense(kStep2TcgenM, kStep2TcgenN);
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int slice_base = issue * 2;
  const uint8_t* a_issue = smem_A_tcgen + slice_base * 16;
  const uint8_t* b_issue = smem_B_tcgen + slice_base * 16;
  const uint64_t a_desc = step2_make_core_matrix_desc_group8_k128(a_issue);
  const uint64_t b_desc = step2_make_core_matrix_desc_group8_k128(b_issue);
  if (warp_id == 0 && lane == 0) {
    step2_tcgen05_mma_f8f6f4_cta1_ss(
        tmem_base, a_desc, b_desc, kIdesc, enable_input_d);
  }
}

__device__ __forceinline__ void scatter_step2_scaled_tmem_64x8(
    uint32_t scaled_tmem,
    const int* __restrict__ valid_token_idx,
    const float* __restrict__ valid_token_w,
    float* __restrict__ out_acc_dev,
    int row_start,
    int row_tile,
    int n_rows,
    int h_base) {
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  if (warp_id >= 4) return;
  const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
  const int col_parity = (lane >> 1) & 1;
  const int h = h_base + result_row;
  float result_bits[4];
  step2_tcgen05_ld_16x64b_x4(result_bits, scaled_tmem);
  step2_tcgen05_wait_ld_sync();
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    const int rr = 2 * reg + col_parity;
    if (rr < n_rows) {
      const int slot = row_start + row_tile + rr;
      const int tok = valid_token_idx[slot];
      const float token_w = valid_token_w[slot];
      atomicAdd(&out_acc_dev[static_cast<int64_t>(tok) * kStep2Hidden + h],
                token_w * result_bits[reg]);
    }
  }
}

__device__ __forceinline__ void compute_step2_loaded_w2_group_row_tile(
    const uint8_t* __restrict__ smem_w2_tma,
    uint8_t* __restrict__ smem_A_tcgen,
    uint8_t* __restrict__ smem_B_tcgen,
    float* __restrict__ smem_scale_diag,
    const uint8_t* __restrict__ c_perm_q_dev,
    const float* __restrict__ c_perm_scale_dev,
    const float* __restrict__ s2_all_dev,
    const int* __restrict__ expert_offset,
    int expert,
    int row_tile,
    int n_rows,
    int k_group,
    int h_scale_block,
    int64_t s2_expert_base,
    bool used_w2_tma,
    uint32_t partial_tmem_base,
    uint32_t scaled_tmem,
    bool& scaled_has_accum,
    uint64_t* tcgen_mma_barrier,
    int& mma_phase_bit) {
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

#pragma unroll
  for (int k_sub = 0; k_sub < kStep2TmaKGroupBlocks; ++k_sub) {
    const int k_blk = k_group * kStep2TmaKGroupBlocks + k_sub;
    const uint8_t* smem_A_for_mma =
        smem_w2_tma + static_cast<int64_t>(k_sub) * kStep2TcgenACompactBytes;
    if (!used_w2_tma) {
      repack_step2_w2_tma_group_to_a_fp8_64x128(
          smem_A_tcgen, smem_w2_tma, k_sub);
      smem_A_for_mma = smem_A_tcgen;
    }
    const float w_scale =
        s2_all_dev[s2_expert_base + h_scale_block * kStep2IntermediateBlocks + k_blk];
    stage_step2_cperm_q_b_fp8_8x128_update_diag_only(
        smem_B_tcgen, smem_scale_diag, c_perm_q_dev, c_perm_scale_dev,
        expert_offset, expert, row_tile, n_rows, k_blk, w_scale);
    __syncthreads();

#pragma unroll
    for (int issue = 0; issue < kStep2K32IssuesPerBlock; ++issue) {
      step2_issue_mma_64x8x128_f8f6f4_ss(
          smem_A_for_mma, smem_B_tcgen,
          partial_tmem_base + static_cast<uint32_t>(issue * kStep2TcgenN),
          false, issue);
    }
    if (warp_id == 0 && lane == 0) {
      step2_tcgen05_commit_group1(tcgen_mma_barrier);
    }
    if (threadIdx.x == 0) {
      step2_tcgen05_wait_mma_barrier_single(tcgen_mma_barrier, mma_phase_bit);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    }
    __syncthreads();
    mma_phase_bit ^= 1;

    constexpr uint32_t kTf32Idesc =
        step2_make_idesc_tf32_ts_f32_dense(kStep2TcgenM, kStep2TcgenN);
#pragma unroll
    for (int issue = 0; issue < kStep2K32IssuesPerBlock; ++issue) {
      const uint32_t partial_tmem =
          partial_tmem_base + static_cast<uint32_t>(issue * kStep2TcgenN);
      const float* diag = smem_scale_diag + issue * (2 * kStep2TcgenN * 4);
      const uint64_t diag_desc = step2_make_tf32_diag_desc(diag);
      if (warp_id == 0 && lane == 0) {
        step2_tcgen05_mma_tf32_cta1_ts(
            scaled_tmem, partial_tmem, diag_desc, kTf32Idesc,
            scaled_has_accum || issue != 0);
      }
    }
    if (warp_id == 0 && lane == 0) {
      step2_tcgen05_commit_group1(tcgen_mma_barrier);
    }
    if (threadIdx.x == 0) {
      step2_tcgen05_wait_mma_barrier_single(tcgen_mma_barrier, mma_phase_bit);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    }
    __syncthreads();
    mma_phase_bit ^= 1;
    scaled_has_accum = true;
  }
}

template <int KGroupBlocks>
__device__ __forceinline__ void compute_step2_loaded_w2_group_row_tile_batch2_kgroup(
    const uint8_t* __restrict__ smem_w2_tma,
    uint8_t (*__restrict__ smem_B_tcgen)[kStep2TcgenBCompactBytes],
    float (*__restrict__ smem_scale_diag)[kStep2K32IssuesPerBlock * 2 * kStep2TcgenN * 4],
    const uint8_t* __restrict__ c_perm_q_dev,
    const float* __restrict__ c_perm_scale_dev,
    const float* __restrict__ s2_all_dev,
    const int* __restrict__ expert_offset,
    int expert,
    int row_tile,
    int n_rows,
    int k_group,
    int h_scale_block,
    int64_t s2_expert_base,
    uint32_t partial_tmem_base,
    uint32_t scaled_tmem,
    bool& scaled_has_accum,
    uint64_t* tcgen_mma_barrier,
    int& mma_phase_bit) {
  constexpr int kBatchSize = 2;
  constexpr int kDiagElemsPerIssue = 2 * kStep2TcgenN * 4;
  constexpr uint32_t kTf32Idesc =
      step2_make_idesc_tf32_ts_f32_dense(kStep2TcgenM, kStep2TcgenN);
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  zero_step2_tf32_diag_scale(smem_scale_diag[0]);
  zero_step2_tf32_diag_scale(smem_scale_diag[1]);
  __syncthreads();

#pragma unroll
  for (int batch_base = 0; batch_base < KGroupBlocks; batch_base += kBatchSize) {
#pragma unroll
    for (int local = 0; local < kBatchSize; ++local) {
      const int k_sub = batch_base + local;
      const int k_blk = k_group * KGroupBlocks + k_sub;
      const float w_scale =
          s2_all_dev[s2_expert_base + h_scale_block * kStep2IntermediateBlocks + k_blk];
      stage_step2_cperm_q_b_fp8_8x128_update_diag_only(
          smem_B_tcgen[local], smem_scale_diag[local], c_perm_q_dev, c_perm_scale_dev,
          expert_offset, expert, row_tile, n_rows, k_blk, w_scale);
    }
    __syncthreads();

#pragma unroll
    for (int local = 0; local < kBatchSize; ++local) {
      const int k_sub = batch_base + local;
      const uint8_t* smem_A_for_mma =
          smem_w2_tma + static_cast<int64_t>(k_sub) * kStep2TcgenACompactBytes;
#pragma unroll
      for (int issue = 0; issue < kStep2K32IssuesPerBlock; ++issue) {
        const uint32_t partial_tmem =
            partial_tmem_base +
            static_cast<uint32_t>((local * kStep2K32IssuesPerBlock + issue) *
                                  kStep2TcgenN);
        step2_issue_mma_64x8x128_f8f6f4_ss(
            smem_A_for_mma, smem_B_tcgen[local], partial_tmem, false, issue);
      }
    }
    if (warp_id == 0 && lane == 0) {
      step2_tcgen05_commit_group1(tcgen_mma_barrier);
    }
    if (threadIdx.x == 0) {
      step2_tcgen05_wait_mma_barrier_single(tcgen_mma_barrier, mma_phase_bit);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    }
    __syncthreads();
    mma_phase_bit ^= 1;

#pragma unroll
    for (int local = 0; local < kBatchSize; ++local) {
#pragma unroll
      for (int issue = 0; issue < kStep2K32IssuesPerBlock; ++issue) {
        const uint32_t partial_tmem =
            partial_tmem_base +
            static_cast<uint32_t>((local * kStep2K32IssuesPerBlock + issue) *
                                  kStep2TcgenN);
        const float* diag =
            smem_scale_diag[local] + issue * kDiagElemsPerIssue;
        const uint64_t diag_desc = step2_make_tf32_diag_desc(diag);
        if (warp_id == 0 && lane == 0) {
          step2_tcgen05_mma_tf32_cta1_ts(
              scaled_tmem, partial_tmem, diag_desc, kTf32Idesc,
              scaled_has_accum || batch_base != 0 || local != 0 || issue != 0);
        }
      }
    }
    if (warp_id == 0 && lane == 0) {
      step2_tcgen05_commit_group1(tcgen_mma_barrier);
    }
    if (threadIdx.x == 0) {
      step2_tcgen05_wait_mma_barrier_single(tcgen_mma_barrier, mma_phase_bit);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    }
    __syncthreads();
    mma_phase_bit ^= 1;
    scaled_has_accum = true;
  }
}

__device__ __forceinline__ void compute_step2_loaded_w2_group_row_tile_batch2(
    const uint8_t* __restrict__ smem_w2_tma,
    uint8_t (*__restrict__ smem_B_tcgen)[kStep2TcgenBCompactBytes],
    float (*__restrict__ smem_scale_diag)[kStep2K32IssuesPerBlock * 2 * kStep2TcgenN * 4],
    const uint8_t* __restrict__ c_perm_q_dev,
    const float* __restrict__ c_perm_scale_dev,
    const float* __restrict__ s2_all_dev,
    const int* __restrict__ expert_offset,
    int expert,
    int row_tile,
    int n_rows,
    int k_group,
    int h_scale_block,
    int64_t s2_expert_base,
    uint32_t partial_tmem_base,
    uint32_t scaled_tmem,
    bool& scaled_has_accum,
    uint64_t* tcgen_mma_barrier,
    int& mma_phase_bit) {
  compute_step2_loaded_w2_group_row_tile_batch2_kgroup<kStep2TmaKGroupBlocks>(
      smem_w2_tma, smem_B_tcgen, smem_scale_diag, c_perm_q_dev, c_perm_scale_dev,
      s2_all_dev, expert_offset, expert, row_tile, n_rows, k_group, h_scale_block,
      s2_expert_base, partial_tmem_base, scaled_tmem, scaled_has_accum,
      tcgen_mma_barrier, mma_phase_bit);
}

// Main experimental Step2 path:
// 1. load W2 in 64x1024 K-groups; sw128 TMA lands directly in tcgen A layout,
// 2. fallback global loads keep the row-major group + repack path,
// 3. stage prequant C as tcgen B and fold W2*C scales through TF32 TMEM MMA,
// 4. scatter the final H tile back to token-major output.
static __global__ void step2_gemm2_scatter_all_experts_tcgen_kernel(
    const int* __restrict__ expert_t_valid,
    const int* __restrict__ expert_offset, const int* __restrict__ valid_token_idx,
    const float* __restrict__ valid_token_w, const int* __restrict__ active_experts,
    const uint8_t* __restrict__ c_perm_q_dev, const float* __restrict__ c_perm_scale_dev,
    const uint8_t* __restrict__ w2_all_dev, const float* __restrict__ s2_all_dev,
    const void* __restrict__ w2_tma_desc,
    float* __restrict__ out_acc_dev) {
  const int h_tile64 = blockIdx.x;
  const int expert = active_experts != nullptr ? active_experts[blockIdx.y] : static_cast<int>(blockIdx.y);
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  if (expert >= kStep2LocalExperts || h_tile64 >= kStep2HiddenTiles64) return;
  const int t_valid = expert_t_valid[expert];
  if (t_valid <= 0) return;
  const int row_start = expert_offset[expert];
  const int h_base = h_tile64 * kStep2TcgenM;
  const int h_scale_block = h_base / kStep2Block;
  const int64_t s2_expert_base =
      static_cast<int64_t>(expert) * kStep2HiddenScaleBlocks * kStep2IntermediateBlocks;

  alignas(1024) __shared__ uint8_t smem_A_tcgen[kStep2TcgenACompactBytes];
  alignas(1024) __shared__ uint8_t smem_B_tcgen[2][kStep2TcgenBCompactBytes];
  alignas(1024) __shared__ uint8_t smem_w2_tma[2][kStep2W2TmaGroupBytes];
  alignas(16) __shared__ float
      smem_scale_diag[2][kStep2K32IssuesPerBlock * 2 * kStep2TcgenN * 4];

#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) && \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  alignas(16) __shared__ uint64_t tcgen_mma_barrier;
  alignas(16) __shared__ uint64_t w2_tma_barrier[2];
  alignas(16) __shared__ uint32_t tcgen_tmem_base_ptr;
  if (threadIdx.x == 0) {
    step2_tcgen05_mbarrier_init(&tcgen_mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp_id == 0) {
    const uint32_t taddr = step2_tcgen05_alloc_cta1_cols_128(&tcgen_tmem_base_ptr);
    if (lane == 0) {
      tcgen_tmem_base_ptr = taddr;
    }
  }
  __syncthreads();

  const uint32_t partial_tmem_base = tcgen_tmem_base_ptr;
  const uint32_t scaled_tmem = tcgen_tmem_base_ptr + 64;

  const bool use_w2_tma = issue_step2_w2_tma_sw128_64x1024(
      smem_w2_tma[0], w2_tma_desc, expert, h_tile64, 0, &w2_tma_barrier[0]);

  for (int k_group = 0; k_group < kStep2TmaKGroups; ++k_group) {
    const int cur = k_group & 1;
    const int next = cur ^ 1;
    if (use_w2_tma) {
      wait_step2_w2_tma(&w2_tma_barrier[cur]);
      if (k_group + 1 < kStep2TmaKGroups) {
        issue_step2_w2_tma_sw128_64x1024(
            smem_w2_tma[next], w2_tma_desc, expert, h_tile64, k_group + 1,
            &w2_tma_barrier[next]);
      }
    } else {
      load_step2_w2_group_direct_rowmajor(
          smem_w2_tma[cur], w2_all_dev, expert, h_tile64, k_group);
      __syncthreads();
    }

    for (int row_tile = 0; row_tile < t_valid; row_tile += kStep2TcgenN) {
      const int n_rows = step2_min_int(t_valid - row_tile, kStep2TcgenN);
      int mma_phase_bit = 0;
      bool scaled_has_accum = false;

      if (use_w2_tma) {
        compute_step2_loaded_w2_group_row_tile_batch2(
            smem_w2_tma[cur], smem_B_tcgen, smem_scale_diag,
            c_perm_q_dev, c_perm_scale_dev, s2_all_dev, expert_offset,
            expert, row_tile, n_rows, k_group, h_scale_block, s2_expert_base,
            partial_tmem_base, scaled_tmem, scaled_has_accum,
            &tcgen_mma_barrier, mma_phase_bit);
      } else {
        zero_step2_tf32_diag_scale(smem_scale_diag[0]);
        __syncthreads();
        compute_step2_loaded_w2_group_row_tile(
            smem_w2_tma[cur], smem_A_tcgen, smem_B_tcgen[0], smem_scale_diag[0],
            c_perm_q_dev, c_perm_scale_dev, s2_all_dev, expert_offset,
            expert, row_tile, n_rows, k_group, h_scale_block, s2_expert_base,
            false, partial_tmem_base, scaled_tmem, scaled_has_accum,
            &tcgen_mma_barrier, mma_phase_bit);
      }
      scatter_step2_scaled_tmem_64x8(
          scaled_tmem, valid_token_idx, valid_token_w, out_acc_dev,
          row_start, row_tile, n_rows, h_base);
      __syncthreads();
    }
  }

  if (warp_id == 0) {
    step2_tcgen05_dealloc_cta1_cols_128(tcgen_tmem_base_ptr);
  }
  __syncthreads();
  if (warp_id == 0) {
    step2_tcgen05_relinquish_alloc_permit_cta1();
  }
#else
  (void)c_perm_q_dev;
  (void)c_perm_scale_dev;
  (void)expert_t_valid;
  (void)expert_offset;
  (void)valid_token_idx;
  (void)valid_token_w;
  (void)w2_all_dev;
  (void)s2_all_dev;
  (void)out_acc_dev;
#endif
}

template <int KGroupBlocks>
static __global__ void step2_gemm2_scatter_all_experts_tcgen_kgroup_kernel(
    const int* __restrict__ expert_t_valid,
    const int* __restrict__ expert_offset, const int* __restrict__ valid_token_idx,
    const float* __restrict__ valid_token_w, const int* __restrict__ active_experts,
    const uint8_t* __restrict__ c_perm_q_dev, const float* __restrict__ c_perm_scale_dev,
    const uint8_t* __restrict__ w2_all_dev, const float* __restrict__ s2_all_dev,
    const void* __restrict__ w2_tma_desc,
    float* __restrict__ out_acc_dev) {
  static_assert(kStep2IntermediateBlocks % KGroupBlocks == 0,
                "Step2 K group must divide K128 blocks.");
  constexpr int kTmaKGroups = kStep2IntermediateBlocks / KGroupBlocks;
  constexpr int kW2TmaGroupBytes = kStep2TcgenM * kStep2Block * KGroupBlocks;

  const int h_tile64 = blockIdx.x;
  const int expert =
      active_experts != nullptr ? active_experts[blockIdx.y] : static_cast<int>(blockIdx.y);
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  if (expert >= kStep2LocalExperts || h_tile64 >= kStep2HiddenTiles64) return;
  const int t_valid = expert_t_valid[expert];
  if (t_valid <= 0) return;
  const int row_start = expert_offset[expert];
  const int h_base = h_tile64 * kStep2TcgenM;
  const int h_scale_block = h_base / kStep2Block;
  const int64_t s2_expert_base =
      static_cast<int64_t>(expert) * kStep2HiddenScaleBlocks * kStep2IntermediateBlocks;

  alignas(1024) __shared__ uint8_t smem_A_tcgen[kStep2TcgenACompactBytes];
  alignas(1024) __shared__ uint8_t smem_B_tcgen[2][kStep2TcgenBCompactBytes];
  alignas(1024) __shared__ uint8_t smem_w2_tma[2][kW2TmaGroupBytes];
  alignas(16) __shared__ float
      smem_scale_diag[2][kStep2K32IssuesPerBlock * 2 * kStep2TcgenN * 4];

#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) && \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  alignas(16) __shared__ uint64_t tcgen_mma_barrier;
  alignas(16) __shared__ uint64_t w2_tma_barrier[2];
  alignas(16) __shared__ uint32_t tcgen_tmem_base_ptr;
  if (threadIdx.x == 0) {
    step2_tcgen05_mbarrier_init(&tcgen_mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp_id == 0) {
    const uint32_t taddr = step2_tcgen05_alloc_cta1_cols_128(&tcgen_tmem_base_ptr);
    if (lane == 0) {
      tcgen_tmem_base_ptr = taddr;
    }
  }
  __syncthreads();

  const uint32_t partial_tmem_base = tcgen_tmem_base_ptr;
  const uint32_t scaled_tmem = tcgen_tmem_base_ptr + 64;
  const bool use_w2_tma = issue_step2_w2_tma_sw128_64xkgroup<KGroupBlocks>(
      smem_w2_tma[0], w2_tma_desc, expert, h_tile64, 0, &w2_tma_barrier[0]);

  for (int k_group = 0; k_group < kTmaKGroups; ++k_group) {
    const int cur = k_group & 1;
    const int next = cur ^ 1;
    if (use_w2_tma) {
      wait_step2_w2_tma(&w2_tma_barrier[cur]);
      if (k_group + 1 < kTmaKGroups) {
        issue_step2_w2_tma_sw128_64xkgroup<KGroupBlocks>(
            smem_w2_tma[next], w2_tma_desc, expert, h_tile64, k_group + 1,
            &w2_tma_barrier[next]);
      }
    } else {
      load_step2_w2_group_direct_rowmajor_kgroup<KGroupBlocks>(
          smem_w2_tma[cur], w2_all_dev, expert, h_tile64, k_group);
      __syncthreads();
    }

    for (int row_tile = 0; row_tile < t_valid; row_tile += kStep2TcgenN) {
      const int n_rows = step2_min_int(t_valid - row_tile, kStep2TcgenN);
      int mma_phase_bit = 0;
      bool scaled_has_accum = false;

      if (use_w2_tma) {
        compute_step2_loaded_w2_group_row_tile_batch2_kgroup<KGroupBlocks>(
            smem_w2_tma[cur], smem_B_tcgen, smem_scale_diag,
            c_perm_q_dev, c_perm_scale_dev, s2_all_dev, expert_offset,
            expert, row_tile, n_rows, k_group, h_scale_block, s2_expert_base,
            partial_tmem_base, scaled_tmem, scaled_has_accum,
            &tcgen_mma_barrier, mma_phase_bit);
      } else if constexpr (KGroupBlocks == kStep2TmaKGroupBlocks) {
        zero_step2_tf32_diag_scale(smem_scale_diag[0]);
        __syncthreads();
        compute_step2_loaded_w2_group_row_tile(
            smem_w2_tma[cur], smem_A_tcgen, smem_B_tcgen[0], smem_scale_diag[0],
            c_perm_q_dev, c_perm_scale_dev, s2_all_dev, expert_offset,
            expert, row_tile, n_rows, k_group, h_scale_block, s2_expert_base,
            false, partial_tmem_base, scaled_tmem, scaled_has_accum,
            &tcgen_mma_barrier, mma_phase_bit);
      } else {
        // Non-TMA fallback remains on the K8 path in the host launcher.
      }
      scatter_step2_scaled_tmem_64x8(
          scaled_tmem, valid_token_idx, valid_token_w, out_acc_dev,
          row_start, row_tile, n_rows, h_base);
      __syncthreads();
    }
  }

  if (warp_id == 0) {
    step2_tcgen05_dealloc_cta1_cols_128(tcgen_tmem_base_ptr);
  }
  __syncthreads();
  if (warp_id == 0) {
    step2_tcgen05_relinquish_alloc_permit_cta1();
  }
#else
  (void)c_perm_q_dev;
  (void)c_perm_scale_dev;
  (void)expert_t_valid;
  (void)expert_offset;
  (void)valid_token_idx;
  (void)valid_token_w;
  (void)w2_all_dev;
  (void)s2_all_dev;
  (void)out_acc_dev;
#endif
}

static inline cudaError_t LaunchStep2DirectAllExperts(
    const int* expert_t_valid, const int* expert_offset,
    const int* valid_token_idx, const float* valid_token_w, const int* active_experts,
    int active_expert_count,
    const uint8_t* c_perm_q_dev, const float* c_perm_scale_dev,
    const uint8_t* w2_all_dev, const float* s2_all_dev,
    const void* w2_tma_desc, float* out_acc_dev, cudaStream_t stream,
    int pipeline_variant) {
  const int grid_experts =
      active_expert_count > 0 ? active_expert_count : kStep2LocalExperts;
  dim3 grid(kStep2HiddenTiles64, grid_experts);
  const bool can_use_k4 = w2_tma_desc != nullptr;
  switch (pipeline_variant) {
    case kStep2PipeM64N8K4DbBatch2:
    case kStep2PipeM64N16K4DbBatch2:
      // The N16 selector is an alias for the real K4 experiment until a wider
      // TMEM layout is added for true N16 partial+scaled accumulation.
      if (can_use_k4) {
        step2_gemm2_scatter_all_experts_tcgen_kgroup_kernel<4>
            <<<grid, dim3(kStep2Threads), 0, stream>>>(
                expert_t_valid, expert_offset, valid_token_idx, valid_token_w, active_experts,
                c_perm_q_dev, c_perm_scale_dev, w2_all_dev, s2_all_dev, w2_tma_desc, out_acc_dev);
      } else {
        step2_gemm2_scatter_all_experts_tcgen_kgroup_kernel<kStep2TmaKGroupBlocks>
            <<<grid, dim3(kStep2Threads), 0, stream>>>(
                expert_t_valid, expert_offset, valid_token_idx, valid_token_w, active_experts,
                c_perm_q_dev, c_perm_scale_dev, w2_all_dev, s2_all_dev, nullptr, out_acc_dev);
      }
      break;
    case kStep2PipeM64N16K8DbBatch2:
    case kStep2PipeM128N8K8DbBatch2:
    case kStep2PipeM64N8K8DbBatch2:
    default:
      // N16/M128 selectors alias to the safe K8 path for now. True N16/M128
      // variants need a separate TMEM/readback/shared-memory design.
      step2_gemm2_scatter_all_experts_tcgen_kgroup_kernel<kStep2TmaKGroupBlocks>
          <<<grid, dim3(kStep2Threads), 0, stream>>>(
              expert_t_valid, expert_offset, valid_token_idx, valid_token_w, active_experts,
              c_perm_q_dev, c_perm_scale_dev, w2_all_dev, s2_all_dev, w2_tma_desc, out_acc_dev);
      break;
  }
  return cudaGetLastError();
}

static inline cudaError_t LaunchStep2DirectAllExperts(
    const int* expert_t_valid, const int* expert_offset,
    const int* valid_token_idx, const float* valid_token_w, const int* active_experts,
    int active_expert_count,
    const uint8_t* c_perm_q_dev, const float* c_perm_scale_dev,
    const uint8_t* w2_all_dev, const float* s2_all_dev,
    const void* w2_tma_desc, float* out_acc_dev, cudaStream_t stream) {
  return LaunchStep2DirectAllExperts(
      expert_t_valid, expert_offset, valid_token_idx, valid_token_w, active_experts,
      active_expert_count, c_perm_q_dev, c_perm_scale_dev, w2_all_dev, s2_all_dev,
      w2_tma_desc, out_acc_dev, stream, kStep2PipeM64N8K8DbBatch2);
}

static inline cudaError_t LaunchStep2DirectAllExperts(
    const float* c_perm_all_dev, const int* expert_t_valid, const int* expert_offset,
    const int* valid_token_idx, const float* valid_token_w, const uint8_t* w2_all_dev,
    const float* s2_all_dev, const void* w2_tma_desc, float* out_acc_dev, cudaStream_t stream) {
  (void)w2_tma_desc;
  constexpr int kThreads = 128;
  for (int expert = 0; expert < kStep2LocalExperts; ++expert) {
    cudaError_t st = cudaPeekAtLastError();
    if (st != cudaSuccess) return st;
    // Compatibility path for callers that have not allocated Step2 prequant buffers.
    // It keeps the original per-expert scalar GEMM2 implementation available.
    int n_rows_host = 0;
    st = cudaMemcpyAsync(
        &n_rows_host, expert_t_valid + expert, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (st != cudaSuccess) return st;
    st = cudaStreamSynchronize(stream);
    if (st != cudaSuccess) return st;
    if (n_rows_host <= 0) continue;
    int row_start_host = 0;
    st = cudaMemcpyAsync(
        &row_start_host, expert_offset + expert, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (st != cudaSuccess) return st;
    st = cudaStreamSynchronize(stream);
    if (st != cudaSuccess) return st;
    const uint8_t* w2_e =
        w2_all_dev + static_cast<size_t>(expert) * kStep2Hidden * kStep2Intermediate;
    const float* s2_e =
        s2_all_dev + static_cast<size_t>(expert) * kStep2HiddenScaleBlocks *
                         kStep2IntermediateBlocks;
    int64_t n_out = static_cast<int64_t>(n_rows_host) * kStep2Hidden;
    int blocks = static_cast<int>((n_out + kThreads - 1) / kThreads);
    step2_gemm2_scatter_direct_kernel<<<blocks, kThreads, 0, stream>>>(
        c_perm_all_dev + static_cast<int64_t>(row_start_host) * kStep2Intermediate,
        n_rows_host, valid_token_idx + row_start_host, valid_token_w + row_start_host,
        w2_e, s2_e, out_acc_dev);
  }
  return cudaGetLastError();
}

}  // namespace direct_backend
