#include "moe_tc_backend.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <memory>

namespace mxfp {

namespace {

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

constexpr int kTile = 128;
constexpr int kRowsPerCta = 16;
constexpr int kThreads = 256;
constexpr int kLanesPerRow = kThreads / kRowsPerCta;  // 16

// TODO(B200): Replace with real TMA copy + tcgen05 warpgroup GEMM1 path.
__global__ void step1_gemm1_swiglu_fused_direct_kernel(const float* __restrict__ a_dev, int hidden,
                                                       int intermediate, int block,
                                                       int hidden_blocks, int intermediate_blocks,
                                                       int n_rows,
                                                       const int* __restrict__ permuted_tok_e,
                                                       const uint8_t* __restrict__ w13_e,
                                                       const float* __restrict__ s13_e,
                                                       float* __restrict__ c_perm_dev) {
  int row = blockIdx.x * kRowsPerCta + (threadIdx.x / kLanesPerRow);
  int lane = threadIdx.x % kLanesPerRow;
  int ib = blockIdx.y;  // [0, I/128)
  if (ib >= intermediate_blocks || row >= n_rows) return;

  int tok = permuted_tok_e[row];
  const float* a_row = a_dev + static_cast<int64_t>(tok) * hidden;
  float* c_row = c_perm_dev + static_cast<int64_t>(row) * intermediate;

  // Register-resident FP32 accumulators keep numerics stable on 5090 temp path.
  // TODO(B200): Revisit accumulator placement when switching to tcgen05 fragments.
  for (int i_chunk = 0; i_chunk < (kTile / kLanesPerRow); ++i_chunk) {
    int i_local = lane + i_chunk * kLanesPerRow;
    int i = ib * block + i_local;
    int j1 = i;
    int j2 = i + intermediate;

    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (int hb = 0; hb < hidden_blocks; ++hb) {
      float scale1 = s13_e[static_cast<int64_t>(ib) * hidden_blocks + hb];
      float scale2 =
          s13_e[static_cast<int64_t>(ib + intermediate_blocks) * hidden_blocks + hb];

      int h0 = hb * block;
      float block_raw1 = 0.0f;
      float block_raw2 = 0.0f;
      for (int u = 0; u < block; ++u) {
        int h = h0 + u;
        float av = a_row[h];
        float w1 = fp8_e4m3fn_to_float_device(w13_e[static_cast<int64_t>(j1) * hidden + h]);
        float w2 = fp8_e4m3fn_to_float_device(w13_e[static_cast<int64_t>(j2) * hidden + h]);
        block_raw1 += av * w1;
        block_raw2 += av * w2;
      }
      acc1 += block_raw1 * scale1;
      acc2 += block_raw2 * scale2;
    }

    c_row[i] = acc1 * siluf_device(acc2);
  }
}

// TODO(B200): Replace staged loads with real TMA bulk tensor copy.
__global__ void step1_gemm1_swiglu_fused_tma_stage_kernel(
    const float* __restrict__ a_dev, int hidden, int intermediate, int block, int hidden_blocks,
    int intermediate_blocks, int n_rows, const int* __restrict__ permuted_tok_e,
    const uint8_t* __restrict__ w13_e, const float* __restrict__ s13_e,
    float* __restrict__ c_perm_dev) {
  extern __shared__ unsigned char smem_u8[];
  float* smem_a = reinterpret_cast<float*>(smem_u8);  // [kRowsPerCta, 128]
  uint8_t* smem_w1 = reinterpret_cast<uint8_t*>(smem_a + kRowsPerCta * kTile);  // [128, 128]
  uint8_t* smem_w2 = smem_w1 + kTile * kTile;                                    // [128, 128]

  int tid = threadIdx.x;
  int row_local = tid / kLanesPerRow;
  int lane = tid % kLanesPerRow;
  int row_base = blockIdx.x * kRowsPerCta;
  int row = row_base + row_local;
  int ib = blockIdx.y;
  if (ib >= intermediate_blocks) return;

  for (int i_chunk = 0; i_chunk < (kTile / kLanesPerRow); ++i_chunk) {
    int i_local = lane + i_chunk * kLanesPerRow;
    int i = ib * block + i_local;
    int j1 = i;
    int j2 = i + intermediate;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    for (int hb = 0; hb < hidden_blocks; ++hb) {
      int h0 = hb * block;

      for (int idx = tid; idx < kRowsPerCta * block; idx += kThreads) {
        int rl = idx / block;
        int u = idx - rl * block;
        int r = row_base + rl;
        float v = 0.0f;
        if (r < n_rows) {
          int tok = permuted_tok_e[r];
          v = a_dev[static_cast<int64_t>(tok) * hidden + h0 + u];
        }
        smem_a[idx] = v;
      }

      for (int idx = tid; idx < block * block; idx += kThreads) {
        int n = idx / block;
        int u = idx - n * block;
        int h = h0 + u;
        int jw1 = ib * block + n;
        int jw2 = jw1 + intermediate;
        smem_w1[idx] = w13_e[static_cast<int64_t>(jw1) * hidden + h];
        smem_w2[idx] = w13_e[static_cast<int64_t>(jw2) * hidden + h];
      }
      __syncthreads();

      if (row < n_rows) {
        float scale1 = s13_e[static_cast<int64_t>(ib) * hidden_blocks + hb];
        float scale2 = s13_e[static_cast<int64_t>(ib + intermediate_blocks) * hidden_blocks + hb];
        float block_raw1 = 0.0f;
        float block_raw2 = 0.0f;
        int a_off = row_local * block;
        int w_off = i_local * block;
        for (int u = 0; u < block; ++u) {
          float av = smem_a[a_off + u];
          block_raw1 += av * fp8_e4m3fn_to_float_device(smem_w1[w_off + u]);
          block_raw2 += av * fp8_e4m3fn_to_float_device(smem_w2[w_off + u]);
        }
        acc1 += block_raw1 * scale1;
        acc2 += block_raw2 * scale2;
      }
      __syncthreads();
    }

    if (row < n_rows) {
      c_perm_dev[static_cast<int64_t>(row) * intermediate + i] = acc1 * siluf_device(acc2);
    }
  }
}

// TODO(B200): Replace with blockwise FP8 GEMM2 Tensor Core path.
__global__ void step2_gemm2_scatter_direct_kernel(const float* __restrict__ c_perm_dev, int hidden,
                                                  int intermediate, int block,
                                                  int intermediate_blocks, int n_rows,
                                                  const int* __restrict__ permuted_tok_e,
                                                  const float* __restrict__ permuted_w_e,
                                                  const uint8_t* __restrict__ w2_e,
                                                  const float* __restrict__ s2_e,
                                                  float* __restrict__ out_acc_dev) {
  int row = blockIdx.x * kRowsPerCta + (threadIdx.x / kLanesPerRow);
  int lane = threadIdx.x % kLanesPerRow;
  int hb = blockIdx.y;  // [0, H/128)
  if (row >= n_rows) return;

  int tok = permuted_tok_e[row];
  float w_tok = permuted_w_e[row];
  const float* c_row = c_perm_dev + static_cast<int64_t>(row) * intermediate;

  for (int h_chunk = 0; h_chunk < (kTile / kLanesPerRow); ++h_chunk) {
    int h_local = lane + h_chunk * kLanesPerRow;
    int h = hb * block + h_local;
    if (h >= hidden) continue;

    float acc = 0.0f;
    for (int ib = 0; ib < intermediate_blocks; ++ib) {
      float scale = s2_e[static_cast<int64_t>(hb) * intermediate_blocks + ib];
      int i0 = ib * block;
      float block_raw = 0.0f;
      for (int u = 0; u < block; ++u) {
        int i = i0 + u;
        float cv = c_row[i];
        float wv = fp8_e4m3fn_to_float_device(w2_e[static_cast<int64_t>(h) * intermediate + i]);
        block_raw += cv * wv;
      }
      acc += block_raw * scale;
    }

    out_acc_dev[static_cast<int64_t>(tok) * hidden + h] += w_tok * acc;
  }
}

// TODO(B200): Replace staged loads with real TMA bulk tensor copy.
__global__ void step2_gemm2_scatter_tma_stage_kernel(
    const float* __restrict__ c_perm_dev, int hidden, int intermediate, int block,
    int intermediate_blocks, int n_rows, const int* __restrict__ permuted_tok_e,
    const float* __restrict__ permuted_w_e, const uint8_t* __restrict__ w2_e,
    const float* __restrict__ s2_e, float* __restrict__ out_acc_dev) {
  extern __shared__ unsigned char smem_u8[];
  float* smem_c = reinterpret_cast<float*>(smem_u8);  // [kRowsPerCta, 128]
  uint8_t* smem_w = reinterpret_cast<uint8_t*>(smem_c + kRowsPerCta * kTile);  // [128, 128]

  int tid = threadIdx.x;
  int row_local = tid / kLanesPerRow;
  int lane = tid % kLanesPerRow;
  int row_base = blockIdx.x * kRowsPerCta;
  int row = row_base + row_local;
  int hb = blockIdx.y;
  if (hb >= hidden / block) return;

  float acc[kTile / kLanesPerRow] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int ib = 0; ib < intermediate_blocks; ++ib) {
    int i0 = ib * block;

    for (int idx = tid; idx < kRowsPerCta * block; idx += kThreads) {
      int rl = idx / block;
      int u = idx - rl * block;
      int r = row_base + rl;
      float v = 0.0f;
      if (r < n_rows) v = c_perm_dev[static_cast<int64_t>(r) * intermediate + i0 + u];
      smem_c[idx] = v;
    }

    for (int idx = tid; idx < block * block; idx += kThreads) {
      int h_local = idx / block;
      int u = idx - h_local * block;
      int h = hb * block + h_local;
      int i = i0 + u;
      smem_w[idx] = w2_e[static_cast<int64_t>(h) * intermediate + i];
    }
    __syncthreads();

    if (row < n_rows) {
      float scale_base = s2_e[static_cast<int64_t>(hb) * intermediate_blocks + ib];
      int a_off = row_local * block;
      for (int h_chunk = 0; h_chunk < (kTile / kLanesPerRow); ++h_chunk) {
        int h_local = lane + h_chunk * kLanesPerRow;
        int w_off = h_local * block;
        float block_raw = 0.0f;
        for (int u = 0; u < block; ++u) {
          block_raw += smem_c[a_off + u] * fp8_e4m3fn_to_float_device(smem_w[w_off + u]);
        }
        acc[h_chunk] += block_raw * scale_base;
      }
    }
    __syncthreads();
  }

  if (row < n_rows) {
    int tok = permuted_tok_e[row];
    float w_tok = permuted_w_e[row];
    for (int h_chunk = 0; h_chunk < (kTile / kLanesPerRow); ++h_chunk) {
      int h_local = lane + h_chunk * kLanesPerRow;
      int h = hb * block + h_local;
      out_acc_dev[static_cast<int64_t>(tok) * hidden + h] += w_tok * acc[h_chunk];
    }
  }
}

class MoeTcBackend5090Temp final : public MoeTcBackend {
 public:
  explicit MoeTcBackend5090Temp(const MoeTcBackendConfig& cfg)
      : cfg_(cfg), available_(false), use_tma_stage_(false) {
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
      available_ = true;
    }
    const char* env_tma = std::getenv("FIB_MOE_TC5090_TMA");
    use_tma_stage_ = (env_tma != nullptr && env_tma[0] == '1');
    if (use_tma_stage_) {
      std::fprintf(stderr,
                   "[mxfp][5090-temp] FIB_MOE_TC5090_TMA=1: using TMA-style staged copy kernels "
                   "(TODO(B200): switch to real TMA ops)\n");
    }
  }

  bool IsAvailable() const override { return available_; }
  const char* BackendName() const override {
    return use_tma_stage_ ? "impl_5090_temp_tma_stage" : "impl_5090_temp_cuda_core";
  }

  cudaError_t RunStep1Fused(const float* a_dev, int n_rows, const int* permuted_tok_e,
                            const uint8_t* w13_e, const float* s13_e,
                            float* c_perm_dev, cudaStream_t stream) override {
    if (!available_ || n_rows <= 0) return available_ ? cudaSuccess : cudaErrorNotSupported;
    dim3 grid((n_rows + kRowsPerCta - 1) / kRowsPerCta, cfg_.intermediate_blocks);
    if (use_tma_stage_) {
      size_t smem = static_cast<size_t>(kRowsPerCta * kTile * sizeof(float)) +
                    static_cast<size_t>(2 * kTile * kTile * sizeof(uint8_t));
      step1_gemm1_swiglu_fused_tma_stage_kernel<<<grid, kThreads, smem, stream>>>(
          a_dev, cfg_.hidden, cfg_.intermediate, cfg_.block, cfg_.hidden_blocks,
          cfg_.intermediate_blocks, n_rows, permuted_tok_e, w13_e, s13_e, c_perm_dev);
      cudaError_t st = cudaPeekAtLastError();
      if (st == cudaSuccess) return st;
      std::fprintf(stderr,
                   "[mxfp][5090-temp] staged Step1 launch failed (%d); falling back to direct\n",
                   static_cast<int>(st));
      cudaGetLastError();
    }
    step1_gemm1_swiglu_fused_direct_kernel<<<grid, kThreads, 0, stream>>>(
        a_dev, cfg_.hidden, cfg_.intermediate, cfg_.block, cfg_.hidden_blocks,
        cfg_.intermediate_blocks, n_rows, permuted_tok_e, w13_e, s13_e, c_perm_dev);
    return cudaPeekAtLastError();
  }

  cudaError_t RunStep2(const float* c_perm_dev, int n_rows, const int* permuted_tok_e,
                       const float* permuted_w_e, const uint8_t* w2_e, const float* s2_e,
                       float* out_acc_dev, cudaStream_t stream) override {
    if (!available_ || n_rows <= 0) return available_ ? cudaSuccess : cudaErrorNotSupported;
    dim3 grid((n_rows + kRowsPerCta - 1) / kRowsPerCta, cfg_.hidden_blocks);
    if (use_tma_stage_) {
      size_t smem = static_cast<size_t>(kRowsPerCta * kTile * sizeof(float)) +
                    static_cast<size_t>(kTile * kTile * sizeof(uint8_t));
      step2_gemm2_scatter_tma_stage_kernel<<<grid, kThreads, smem, stream>>>(
          c_perm_dev, cfg_.hidden, cfg_.intermediate, cfg_.block, cfg_.intermediate_blocks,
          n_rows, permuted_tok_e, permuted_w_e, w2_e, s2_e, out_acc_dev);
      cudaError_t st = cudaPeekAtLastError();
      if (st == cudaSuccess) return st;
      std::fprintf(stderr,
                   "[mxfp][5090-temp] staged Step2 launch failed (%d); falling back to direct\n",
                   static_cast<int>(st));
      cudaGetLastError();
    }
    step2_gemm2_scatter_direct_kernel<<<grid, kThreads, 0, stream>>>(
        c_perm_dev, cfg_.hidden, cfg_.intermediate, cfg_.block, cfg_.intermediate_blocks, n_rows,
        permuted_tok_e, permuted_w_e, w2_e, s2_e, out_acc_dev);
    return cudaPeekAtLastError();
  }

 private:
  MoeTcBackendConfig cfg_;
  bool available_;
  bool use_tma_stage_;
};

}  // namespace

std::unique_ptr<MoeTcBackend> CreateMoeTcBackend5090Temp(const MoeTcBackendConfig& cfg) {
  return std::unique_ptr<MoeTcBackend>(new MoeTcBackend5090Temp(cfg));
}

}  // namespace mxfp
