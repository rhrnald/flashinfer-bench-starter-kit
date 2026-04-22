#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace direct_backend {

static constexpr int kStep1Hidden = 7168;
static constexpr int kStep1Intermediate = 2048;
static constexpr int kStep1Gemm1Out = 2 * kStep1Intermediate;  // 4096
static constexpr int kStep1Block = 128;
static constexpr int kStep1LocalExperts = 32;
static constexpr int kStep1HiddenBlocks = kStep1Hidden / kStep1Block;              // 56
static constexpr int kStep1IntermediateBlocks = kStep1Intermediate / kStep1Block;  // 16
static constexpr int kStep1Gemm1OutBlocks = kStep1Gemm1Out / kStep1Block;          // 32
static constexpr int kStep1Threads = 32;
static constexpr int kStep1TmaWaitIters = 8192;  // Reserved for future TMA path.

static_assert(kStep1Hidden % kStep1Block == 0, "Hidden size must be divisible by block size.");
static_assert(kStep1Intermediate % kStep1Block == 0,
              "Intermediate size must be divisible by block size.");

static __host__ __device__ __forceinline__ int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

static __host__ __device__ __forceinline__ std::size_t align_up(std::size_t x, std::size_t a) {
  return (x + a - 1) & ~(a - 1);
}

// -----------------------------------------------------------------------------
// FP8 E4M3FN decode for the temporary direct correctness path.
// -----------------------------------------------------------------------------
static __device__ __forceinline__ float fp8_e4m3fn_to_float_decode(uint8_t x) {
  const int sign = (x & 0x80) ? -1 : 1;
  const int exp = (x >> 3) & 0xF;
  const int mant = x & 0x7;

  if (exp == 0) {
    if (mant == 0) {
      return sign > 0 ? 0.0f : -0.0f;
    }
    return sign * ldexpf(static_cast<float>(mant) / 8.0f, -6);
  }

  // E4M3FN uses exp=0xF, mant=0x7 as NaN.
  if (exp == 0xF && mant == 0x7) {
    return nanf("");
  }

  const float frac = 1.0f + static_cast<float>(mant) / 8.0f;
  return sign * ldexpf(frac, exp - 7);
}

static __device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

// -----------------------------------------------------------------------------
// Dynamic shared-memory size helper.
//
// Layout:
//   [ B_gate | B_up | A | gate_acc | up_acc ]
//
// B_gate:   [128, 128] bytes
// B_up:     [128, 128] bytes
// A:        [T_valid, 128] bytes
// gate_acc: [T_valid, 128] floats
// up_acc:   [T_valid, 128] floats
// -----------------------------------------------------------------------------
static __host__ __device__ __forceinline__ std::size_t step1_smem_bytes(int t_valid) {
  std::size_t offset = 0;

  offset += kStep1Block * kStep1Block;                                        // B_gate
  offset += kStep1Block * kStep1Block;                                        // B_up
  offset += static_cast<std::size_t>(t_valid) * kStep1Block;                  // A
  offset = align_up(offset, alignof(float));
  offset += static_cast<std::size_t>(t_valid) * kStep1Block * sizeof(float);  // gate_acc
  offset += static_cast<std::size_t>(t_valid) * kStep1Block * sizeof(float);  // up_acc

  return offset;
}

// -----------------------------------------------------------------------------
// Direct A load.
//
// Shared layout:
//   smem_A[row * 128 + kk]
// -----------------------------------------------------------------------------
static __device__ __forceinline__ void load_A_direct(
    uint8_t* smem_A,
    const uint8_t* __restrict__ hidden_fp8_dev,
    const int* __restrict__ valid_token_idx,
    int packed_base,
    int t_valid,
    int k_blk) {
  const int k0 = k_blk * kStep1Block;

  for (int linear = threadIdx.x; linear < t_valid * kStep1Block; linear += blockDim.x) {
    const int row = linear / kStep1Block;
    const int kk = linear % kStep1Block;

    const int packed_idx = packed_base + row;
    const int token_idx = valid_token_idx[packed_idx];

    smem_A[row * kStep1Block + kk] =
        hidden_fp8_dev[static_cast<int64_t>(token_idx) * kStep1Hidden + (k0 + kk)];
  }
}

// -----------------------------------------------------------------------------
// Direct B load.
//
// Logical global layout:
//   w13_all_dev[e, out, hidden] = [32, 4096, 7168]
//
// Shared layout:
//   smem_B_gate[kk * 128 + nn] = gate tile [k, n]
//   smem_B_up  [kk * 128 + nn] = up   tile [k, n]
// -----------------------------------------------------------------------------
static __device__ __forceinline__ void load_B_direct(
    uint8_t* smem_B_gate,
    uint8_t* smem_B_up,
    const uint8_t* __restrict__ w13_all_dev,
    int expert,
    int n_tile,
    int k_blk) {
  const int gate_tile = n_tile;
  const int up_tile = n_tile + kStep1IntermediateBlocks;
  const int k0 = k_blk * kStep1Block;

  for (int linear = threadIdx.x; linear < kStep1Block * kStep1Block; linear += blockDim.x) {
    const int kk = linear / kStep1Block;
    const int nn = linear % kStep1Block;

    const int gate_col = gate_tile * kStep1Block + nn;
    const int up_col = up_tile * kStep1Block + nn;
    const int k_idx = k0 + kk;

    smem_B_gate[kk * kStep1Block + nn] =
        w13_all_dev[((expert * kStep1Gemm1Out + gate_col) * kStep1Hidden) + k_idx];

    smem_B_up[kk * kStep1Block + nn] =
        w13_all_dev[((expert * kStep1Gemm1Out + up_col) * kStep1Hidden) + k_idx];
  }
}

// -----------------------------------------------------------------------------
// Reserved interface for future TMA path.
// Keep the same destination shared-memory layout as load_B_direct().
// -----------------------------------------------------------------------------
static __device__ __forceinline__ void load_B_tma(
    uint8_t* smem_B_gate,
    uint8_t* smem_B_up,
    const void* __restrict__ w13_tma_desc,
    int expert,
    int n_tile,
    int k_blk) {
  (void)smem_B_gate;
  (void)smem_B_up;
  (void)w13_tma_desc;
  (void)expert;
  (void)n_tile;
  (void)k_blk;

  // TODO: replace load_B_direct() with TMA later.
}

// -----------------------------------------------------------------------------
// Temporary correctness-first direct kernel.
//
// Current path:
//   - A load: direct gather -> shared memory
//   - B load: direct load -> shared memory
//   - Compute: direct FP32 accumulation in shared memory
//   - Final activation: SwiGLU = silu(gate) * up
//
// Later replacements:
//   - load_B_direct() -> load_B_tma()
//   - direct inner GEMM loop -> tcgen05 path
// -----------------------------------------------------------------------------
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

  const int n_tile = blockIdx.x;  // 0 .. 15
  const int expert = blockIdx.y;  // 0 .. 31

  const int t_valid = expert_t_valid[expert];
  if (t_valid <= 0) return;

  const int packed_base = expert_offset[expert];

  extern __shared__ __align__(16) uint8_t smem_raw[];

  std::size_t offset = 0;

  uint8_t* smem_B_gate = smem_raw + offset;
  offset += kStep1Block * kStep1Block;

  uint8_t* smem_B_up = smem_raw + offset;
  offset += kStep1Block * kStep1Block;

  uint8_t* smem_A = smem_raw + offset;
  offset += static_cast<std::size_t>(t_valid) * kStep1Block;

  offset = align_up(offset, alignof(float));

  float* smem_gate_acc = reinterpret_cast<float*>(smem_raw + offset);
  offset += static_cast<std::size_t>(t_valid) * kStep1Block * sizeof(float);

  float* smem_up_acc = reinterpret_cast<float*>(smem_raw + offset);

  for (int linear = threadIdx.x; linear < t_valid * kStep1Block; linear += blockDim.x) {
    smem_gate_acc[linear] = 0.0f;
    smem_up_acc[linear] = 0.0f;
  }
  __syncthreads();

  for (int k_blk = 0; k_blk < kStep1HiddenBlocks; ++k_blk) {
    load_A_direct(smem_A, hidden_fp8_dev, valid_token_idx, packed_base, t_valid, k_blk);
    load_B_direct(smem_B_gate, smem_B_up, w13_all_dev, expert, n_tile, k_blk);
    __syncthreads();

    const float b_scale_gate =
        s13_all_dev[(expert * kStep1Gemm1OutBlocks + n_tile) * kStep1HiddenBlocks + k_blk];

    const float b_scale_up = s13_all_dev[(expert * kStep1Gemm1OutBlocks +
                                          (n_tile + kStep1IntermediateBlocks)) *
                                             kStep1HiddenBlocks +
                                         k_blk];

    for (int linear = threadIdx.x; linear < t_valid * kStep1Block; linear += blockDim.x) {
      const int row = linear / kStep1Block;
      const int col = linear % kStep1Block;

      const int token_idx = valid_token_idx[packed_base + row];
      const float a_scale = hidden_scale_dev[static_cast<int64_t>(k_blk) * t + token_idx];

      float partial_gate = 0.0f;
      float partial_up = 0.0f;

      for (int kk = 0; kk < kStep1Block; ++kk) {
        const float a = fp8_e4m3fn_to_float_decode(smem_A[row * kStep1Block + kk]);
        const float bg = fp8_e4m3fn_to_float_decode(smem_B_gate[kk * kStep1Block + col]);
        const float bu = fp8_e4m3fn_to_float_decode(smem_B_up[kk * kStep1Block + col]);

        partial_gate += a * bg;
        partial_up += a * bu;
      }

      smem_gate_acc[row * kStep1Block + col] += partial_gate * a_scale * b_scale_gate;
      smem_up_acc[row * kStep1Block + col] += partial_up * a_scale * b_scale_up;
    }

    __syncthreads();
  }

  for (int linear = threadIdx.x; linear < t_valid * kStep1Block; linear += blockDim.x) {
    const int row = linear / kStep1Block;
    const int col = linear % kStep1Block;

    const float gate = smem_gate_acc[row * kStep1Block + col];
    const float up = smem_up_acc[row * kStep1Block + col];
    const float out = silu(gate) * up;

    c_perm_all_dev[(static_cast<int64_t>(packed_base + row) * kStep1Intermediate) +
                   (n_tile * kStep1Block + col)] = out;
  }
}

// -----------------------------------------------------------------------------
// Header-only launch helper.
//
// Notes:
//   - Signature matches the current external call site.
//   - max_t_valid is computed by copying the 32 expert counts to host.
//   - hidden_tma_desc and w13_tma_desc are intentionally passed as nullptr for now.
// -----------------------------------------------------------------------------
static inline cudaError_t RunStep1AllExpertsDirect(
    const uint8_t* hidden_fp8_dev,
    const float* hidden_scale_dev,
    int64_t t,
    const int* expert_t_valid_dev,
    const int* expert_offset_dev,
    const int* valid_token_idx_dev,
    const uint8_t* w13_all_dev,
    const float* s13_all_dev,
    float* c_perm_all_dev,
    cudaStream_t stream) {
  int expert_t_valid_host[kStep1LocalExperts];

  cudaError_t st =
      cudaMemcpy(expert_t_valid_host, expert_t_valid_dev, sizeof(expert_t_valid_host),
                 cudaMemcpyDeviceToHost);
  if (st != cudaSuccess) {
    return st;
  }

  int max_t_valid = 0;
  for (int e = 0; e < kStep1LocalExperts; ++e) {
    if (expert_t_valid_host[e] > max_t_valid) {
      max_t_valid = expert_t_valid_host[e];
    }
  }

  const std::size_t smem_bytes = step1_smem_bytes(max_t_valid);

  int device = 0;
  st = cudaGetDevice(&device);
  if (st != cudaSuccess) {
    return st;
  }

  int max_dynamic_smem = 0;
  st = cudaDeviceGetAttribute(&max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                              device);
  if (st != cudaSuccess) {
    return st;
  }

  if (smem_bytes > static_cast<std::size_t>(max_dynamic_smem)) {
    return cudaErrorInvalidValue;
  }

  st = cudaFuncSetAttribute(step1_gemm1_swiglu_direct_kernel,
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
      expert_t_valid_dev,
      expert_offset_dev,
      valid_token_idx_dev,
      w13_all_dev,
      s13_all_dev,
      nullptr,
      nullptr,
      c_perm_all_dev);

  return cudaGetLastError();
}

}  // namespace direct_backend
