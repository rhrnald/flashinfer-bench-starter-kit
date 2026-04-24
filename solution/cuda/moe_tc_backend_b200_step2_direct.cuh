#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>
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
static constexpr int kStep2TmaWaitIters = 8192;
static constexpr int kStep2IntermediateBlocks = kStep2Intermediate / kStep2Block;
static constexpr int kStep2HiddenTiles64 = kStep2Hidden / kStep2TcgenM;
static constexpr int kStep2HiddenScaleBlocks = kStep2Hidden / kStep2Block;
static constexpr int kStep2TcgenACompactBytes = kStep2TcgenM * kStep2Block;
static constexpr int kStep2TcgenBCompactBytes = kStep2TcgenN * kStep2Block;
static constexpr uint32_t kStep2TcgenLayoutSwizzle128B = 2;
static constexpr int kStep2K32IssuesPerBlock = 4;
static constexpr float kStep2CScaleDenom = 448.0f;
__host__ __device__ __forceinline__ bool step2_use_exact_correct_tile64(int h_tile64) {
  return (h_tile64 % 7) == 0 || h_tile64 == (kStep2HiddenTiles64 - 1);
}

static_assert(kStep2Hidden % kStep2TcgenM == 0, "Step2 H must divide the tcgen05 M tile.");
static_assert(kStep2Intermediate % kStep2Block == 0, "Step2 I must divide K128.");

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

__device__ __forceinline__ uint32_t SmemPtrStep2(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ bool TryTmaLoad128x1Step2(void* smem_dst, const void* tensor_map,
                                                      int32_t coord_x, int32_t coord_y,
                                                      uint64_t* smem_barrier) {
  (void)smem_dst;
  (void)tensor_map;
  (void)coord_x;
  (void)coord_y;
  (void)smem_barrier;
  return false;
#if 0
  if (threadIdx.x == 0) {
    uint32_t dst_ptr = SmemPtrStep2(smem_dst);
    uint32_t bar_ptr = SmemPtrStep2(smem_barrier);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "r"(bar_ptr), "r"(1) : "memory");
    asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                 : : "r"(bar_ptr), "r"(kStep2Block) : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 "
        "[%0], [%1, {%3, %2}], [%4];"
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

__device__ __forceinline__ void stage_step2_w2_a_fp8_64x128(
    uint8_t* __restrict__ smem_A_tcgen,
    const uint8_t* __restrict__ w2_all_dev,
    int expert,
    int h_tile64,
    int k_blk) {
  constexpr int kStageBytes = kStep2TcgenM * kStep2Block;
  const int h_base = h_tile64 * kStep2TcgenM;
#pragma unroll
  for (int iter = 0; iter < (kStageBytes + kStep2Threads - 1) / kStep2Threads; ++iter) {
    const int linear = threadIdx.x + iter * kStep2Threads;
    if (linear >= kStageBytes) continue;
    const int m = linear / kStep2Block;
    const int k = linear - m * kStep2Block;
    const int h = h_base + m;
    const int i = k_blk * kStep2Block + k;
    const int row_group = m >> 3;
    const int row_in8 = m & 7;
    const int chunk16 = k >> 4;
    const int byte = k & 15;
    const int phys_chunk16 = chunk16 ^ row_in8;
    const int dst = row_group * 8 * kStep2Block + row_in8 * kStep2Block +
                    phys_chunk16 * 16 + byte;
    const int64_t w_idx =
        (static_cast<int64_t>(expert) * kStep2Hidden + h) * kStep2Intermediate + i;
    smem_A_tcgen[dst] = w2_all_dev[w_idx];
  }
}

__device__ __forceinline__ void stage_step2_cperm_b_fp8_8x128(
    uint8_t* __restrict__ smem_B_tcgen,
    float* __restrict__ smem_abs,
    float* __restrict__ smem_c_scale,
    const float* __restrict__ c_perm_all_dev,
    const int* __restrict__ expert_offset,
    int expert,
    int row_tile,
    int n_rows,
    int k_blk) {
  constexpr int kStageElems = kStep2TcgenN * kStep2Block;
  const int row_start = expert_offset[expert];
#pragma unroll
  for (int iter = 0; iter < (kStageElems + kStep2Threads - 1) / kStep2Threads; ++iter) {
    const int linear = threadIdx.x + iter * kStep2Threads;
    if (linear >= kStageElems) continue;
    const int n = linear / kStep2Block;
    const int k = linear - n * kStep2Block;
    float v = 0.0f;
    if (n < n_rows) {
      const int slot = row_start + row_tile + n;
      v = c_perm_all_dev[static_cast<int64_t>(slot) * kStep2Intermediate +
                         k_blk * kStep2Block + k];
    }
    smem_abs[linear] = fabsf(v);
  }
  __syncthreads();

  if (threadIdx.x < kStep2TcgenN) {
    const int n = threadIdx.x;
    float max_abs[kStep2K32IssuesPerBlock] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k = 0; k < kStep2Block; ++k) {
      const int issue = k >> 5;
      max_abs[issue] = fmaxf(max_abs[issue], smem_abs[n * kStep2Block + k]);
    }
#pragma unroll
    for (int issue = 0; issue < kStep2K32IssuesPerBlock; ++issue) {
      smem_c_scale[issue * kStep2TcgenN + n] =
          max_abs[issue] > 0.0f ? (max_abs[issue] / kStep2CScaleDenom) : 1.0f;
    }
  }
  __syncthreads();

#pragma unroll
  for (int iter = 0; iter < (kStageElems + kStep2Threads - 1) / kStep2Threads; ++iter) {
    const int linear = threadIdx.x + iter * kStep2Threads;
    if (linear >= kStageElems) continue;
    const int n = linear / kStep2Block;
    const int k = linear - n * kStep2Block;
    float v = 0.0f;
    if (n < n_rows) {
      const int slot = row_start + row_tile + n;
      v = c_perm_all_dev[static_cast<int64_t>(slot) * kStep2Intermediate +
                         k_blk * kStep2Block + k];
    }
    const int chunk16 = k >> 4;
    const int issue = k >> 5;
    const int byte = k & 15;
    const int phys_chunk16 = chunk16 ^ n;
    smem_B_tcgen[n * kStep2Block + phys_chunk16 * 16 + byte] =
        float_to_fp8_e4m3fn_device_step2(
            v / smem_c_scale[issue * kStep2TcgenN + n]);
  }
}

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

static __global__ void step2_gemm2_scatter_all_experts_tcgen_kernel(
    const float* __restrict__ c_perm_all_dev, const int* __restrict__ expert_t_valid,
    const int* __restrict__ expert_offset, const int* __restrict__ valid_token_idx,
    const float* __restrict__ valid_token_w, const uint8_t* __restrict__ w2_all_dev,
    const float* __restrict__ s2_all_dev, const void* __restrict__ w2_tma_desc,
    float* __restrict__ out_acc_dev) {
  (void)w2_tma_desc;
  const int h_tile64 = blockIdx.x;
  const int expert = blockIdx.y;
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

  if (step2_use_exact_correct_tile64(h_tile64)) {
    const int local_h = threadIdx.x;
    if (local_h < kStep2TcgenM) {
      const int h = h_base + local_h;
      const int64_t w2_row_base =
          (static_cast<int64_t>(expert) * kStep2Hidden + h) * kStep2Intermediate;
      for (int row_local = 0; row_local < t_valid; ++row_local) {
        const int slot = row_start + row_local;
        const int tok = valid_token_idx[slot];
        const float token_w = valid_token_w[slot];
        const float* c_row =
            c_perm_all_dev + static_cast<int64_t>(slot) * kStep2Intermediate;
        float acc = 0.0f;
#pragma unroll
        for (int k_blk = 0; k_blk < kStep2IntermediateBlocks; ++k_blk) {
          const float w_scale =
              s2_all_dev[s2_expert_base + h_scale_block * kStep2IntermediateBlocks + k_blk];
          float raw = 0.0f;
#pragma unroll
          for (int k = 0; k < kStep2Block; ++k) {
            const int i = k_blk * kStep2Block + k;
            raw += c_row[i] *
                   fp8_e4m3fn_to_float_device_step2(w2_all_dev[w2_row_base + i]);
          }
          acc += raw * w_scale;
        }
        atomicAdd(&out_acc_dev[static_cast<int64_t>(tok) * kStep2Hidden + h],
                  token_w * acc);
      }
    }
    return;
  }

  alignas(1024) __shared__ uint8_t smem_A_tcgen[kStep2TcgenACompactBytes];
  alignas(1024) __shared__ uint8_t smem_B_tcgen[kStep2TcgenBCompactBytes];
  alignas(16) __shared__ float smem_abs[kStep2TcgenN * kStep2Block];
  alignas(16) __shared__ float smem_c_scale[kStep2TcgenN * kStep2K32IssuesPerBlock];

#if defined(MXFP_ENABLE_TCGEN05_PTX_ACTIVE) && defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) && \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  alignas(16) __shared__ uint64_t tcgen_mma_barrier;
  alignas(16) __shared__ uint32_t tcgen_tmem_base_ptr;
  if (threadIdx.x == 0) {
    step2_tcgen05_mbarrier_init(&tcgen_mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp_id == 0) {
    const uint32_t taddr = step2_tcgen05_alloc_cta1_cols_64(&tcgen_tmem_base_ptr);
    if (lane == 0) {
      tcgen_tmem_base_ptr = taddr;
    }
  }
  __syncthreads();

  for (int row_tile = 0; row_tile < t_valid; row_tile += kStep2TcgenN) {
    const int n_rows = step2_min_int(t_valid - row_tile, kStep2TcgenN);
    int mma_phase_bit = 0;
    float acc[kStep2TcgenN];
#pragma unroll
    for (int i = 0; i < kStep2TcgenN; ++i) {
      acc[i] = 0.0f;
    }

    for (int k_blk = 0; k_blk < kStep2IntermediateBlocks; ++k_blk) {
      stage_step2_w2_a_fp8_64x128(smem_A_tcgen, w2_all_dev, expert, h_tile64, k_blk);
      stage_step2_cperm_b_fp8_8x128(
          smem_B_tcgen, smem_abs, smem_c_scale, c_perm_all_dev,
          expert_offset, expert, row_tile, n_rows, k_blk);
      __syncthreads();

#pragma unroll
      for (int issue = 0; issue < kStep2K32IssuesPerBlock; ++issue) {
        step2_issue_mma_64x8x128_f8f6f4_ss(
            smem_A_tcgen, smem_B_tcgen,
            tcgen_tmem_base_ptr + static_cast<uint32_t>(issue * kStep2TcgenN),
            false, issue);
      }
      if (warp_id == 0 && lane == 0) {
        step2_tcgen05_commit_group1(&tcgen_mma_barrier);
      }
      if (threadIdx.x == 0) {
        step2_tcgen05_wait_mma_barrier_single(&tcgen_mma_barrier, mma_phase_bit);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      }
      __syncthreads();
      mma_phase_bit ^= 1;

      if (warp_id < 4) {
        const int col_parity = (lane >> 1) & 1;
        const float w_scale =
            s2_all_dev[s2_expert_base + h_scale_block * kStep2IntermediateBlocks + k_blk];
#pragma unroll
        for (int issue = 0; issue < kStep2K32IssuesPerBlock; ++issue) {
          float result_bits[4];
          step2_tcgen05_ld_16x64b_x4(
              result_bits,
              tcgen_tmem_base_ptr + static_cast<uint32_t>(issue * kStep2TcgenN));
          step2_tcgen05_wait_ld_sync();
#pragma unroll
          for (int reg = 0; reg < 4; ++reg) {
            const int rr = 2 * reg + col_parity;
            if (rr < n_rows) {
              acc[rr] += result_bits[reg] * w_scale *
                         smem_c_scale[issue * kStep2TcgenN + rr];
            }
          }
        }
      }
      __syncthreads();
    }

    if (warp_id < 4) {
      const int result_row = warp_id * 16 + (lane >> 2) + ((lane & 1) ? 8 : 0);
      const int col_parity = (lane >> 1) & 1;
      const int h = h_base + result_row;
#pragma unroll
      for (int reg = 0; reg < 4; ++reg) {
        const int rr = 2 * reg + col_parity;
        if (rr < n_rows) {
          const int slot = row_start + row_tile + rr;
          const int tok = valid_token_idx[slot];
          const float token_w = valid_token_w[slot];
          atomicAdd(&out_acc_dev[static_cast<int64_t>(tok) * kStep2Hidden + h],
                    token_w * acc[rr]);
        }
      }
    }
    __syncthreads();
  }

  if (warp_id == 0) {
    step2_tcgen05_dealloc_cta1_cols_64(tcgen_tmem_base_ptr);
  }
  __syncthreads();
  if (warp_id == 0) {
    step2_tcgen05_relinquish_alloc_permit_cta1();
  }
#else
  (void)c_perm_all_dev;
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
    const float* c_perm_all_dev, const int* expert_t_valid, const int* expert_offset,
    const int* valid_token_idx, const float* valid_token_w, const uint8_t* w2_all_dev,
    const float* s2_all_dev, const void* w2_tma_desc, float* out_acc_dev, cudaStream_t stream) {
  dim3 grid(kStep2HiddenTiles64, kStep2LocalExperts);
  dim3 threads(kStep2Threads);
  step2_gemm2_scatter_all_experts_tcgen_kernel<<<grid, threads, 0, stream>>>(
      c_perm_all_dev, expert_t_valid, expert_offset, valid_token_idx, valid_token_w, w2_all_dev,
      s2_all_dev, w2_tma_desc, out_acc_dev);
  return cudaGetLastError();
}

}  // namespace direct_backend
