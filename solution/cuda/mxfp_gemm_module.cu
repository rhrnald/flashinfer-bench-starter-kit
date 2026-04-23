#include "mxfp_gemm_module.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define FIB_HAS_CUDA_FP8 1
#else
#define FIB_HAS_CUDA_FP8 0
#endif

#if __has_include(<cutlass/cutlass.h>) && __has_include(<cute/tensor.hpp>) && \
    __has_include(<cutlass/gemm/device/gemm_universal_adapter.h>)
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/detail/sm100_mixed_dtype_blockwise_layout.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/kernel_hardware_info.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>
#define FIB_HAS_DIRECT_CUTLASS_SM100 1
#else
#define FIB_HAS_DIRECT_CUTLASS_SM100 0
#endif

#if __has_include(<flashinfer/gemm/group_gemm_fp8_groupwise_sm100.cuh>)
#include <flashinfer/gemm/group_gemm_fp8_groupwise_sm100.cuh>
#define FIB_HAS_FLASHINFER_FP8_GROUP_GEMM_SM100 1
#else
#define FIB_HAS_FLASHINFER_FP8_GROUP_GEMM_SM100 0
#endif

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <array>
#include <vector>

namespace mxfp {

namespace {

inline float siluf_host(float x) { return x / (1.0f + std::exp(-x)); }

__device__ __forceinline__ float fp8_e4m3fn_to_float_device(uint8_t x) {
  int sign = (x & 0x80) ? -1 : 1;
  int exp = (x >> 3) & 0x0f;
  int mant = x & 0x07;

  if (exp == 0) {
    if (mant == 0) {
      return sign == 1 ? 0.0f : -0.0f;
    }
    float frac = static_cast<float>(mant) * 0.125f;
    return sign * ldexpf(frac, -6);
  }

  float frac = 1.0f + static_cast<float>(mant) * 0.125f;
  return sign * ldexpf(frac, exp - 7);
}

__device__ __forceinline__ float siluf_device(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float bf16_to_float_device(uint16_t bits) {
  uint32_t u32 = static_cast<uint32_t>(bits) << 16;
  return __uint_as_float(u32);
}

__device__ __forceinline__ float f16_to_float_device(uint16_t bits) {
  union {
    uint16_t u;
    __half h;
  } v;
  v.u = bits;
  return __half2float(v.h);
}

__device__ __forceinline__ uint8_t float_to_e4m3_device(float x) {
#if FIB_HAS_CUDA_FP8
  __nv_fp8_e4m3 y(x);
  return *reinterpret_cast<uint8_t*>(&y);
#else
  (void)x;
  return 0;
#endif
}

__device__ __forceinline__ float fp8_native_to_float_device(uint8_t x) {
#if FIB_HAS_CUDA_FP8
  __nv_fp8_e4m3 y = *reinterpret_cast<__nv_fp8_e4m3*>(&x);
  return static_cast<float>(y);
#else
  (void)x;
  return 0.0f;
#endif
}

#if FIB_HAS_DIRECT_CUTLASS_SM100

inline char* align_ptr(char* ptr, size_t alignment) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
  return reinterpret_cast<char*>(aligned);
}

template <typename T>
T* reserve_workspace(char*& cursor, size_t& remaining, size_t count = 1) {
  char* aligned = align_ptr(cursor, 16);
  size_t padding = static_cast<size_t>(aligned - cursor);
  size_t bytes = sizeof(T) * count;
  if (padding + bytes > remaining) {
    return nullptr;
  }
  cursor = aligned + bytes;
  remaining -= padding + bytes;
  return reinterpret_cast<T*>(aligned);
}

inline cudaError_t cutlass_status_to_cuda(cutlass::Status status) {
  return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

template <int ScaleGranularityM, bool ScaleMajorK, int MmaSM, typename DTypeA, typename DTypeB,
          typename DTypeOut>
cudaError_t launch_cutlass_blockscaled_group_gemm_sm100(
    void* arg_buffer, size_t arg_buffer_size_in_bytes, void* workspace,
    size_t workspace_size_in_bytes, DTypeA* a_ptr, DTypeB* b_ptr, float* sfa_ptr,
    float* sfb_ptr, DTypeOut* d_ptr, int m, int n, int k, cudaStream_t stream) {
  using namespace cute;
  using ElementA = DTypeA;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = DTypeB;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = DTypeOut;
  using LayoutD = cutlass::layout::RowMajor;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void;
  using LayoutC = void;
  constexpr int AlignmentC = 0;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using MmaTileShape = Shape<Int<MmaSM * 128>, _128, _128>;
  using ClusterShape = Shape<Int<MmaSM>, _1, _1>;
  using ScaleConfig = std::conditional_t<
      ScaleMajorK,
      cutlass::detail::Sm1xxBlockwiseScaleConfig<ScaleGranularityM, 128, 128, UMMA::Major::K,
                                                 UMMA::Major::K>,
      cutlass::detail::Sm1xxBlockwiseScaleConfig<ScaleGranularityM, 128, 128,
                                                 UMMA::Major::MN, UMMA::Major::MN>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, typename ScaleConfig::LayoutSFA>, AlignmentA, ElementB,
      cute::tuple<LayoutB, typename ScaleConfig::LayoutSFB>, AlignmentB, ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  (void)arg_buffer;
  (void)arg_buffer_size_in_bytes;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {a_ptr, stride_a, b_ptr, stride_b, sfa_ptr, layout_sfa, sfb_ptr, layout_sfb},
      {{}, d_ptr, stride_d, d_ptr, stride_d}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  Gemm gemm;
  size_t required_workspace = Gemm::get_workspace_size(arguments);
  if (required_workspace > workspace_size_in_bytes) {
    return cudaErrorMemoryAllocation;
  }

  cudaError_t status = cutlass_status_to_cuda(gemm.can_implement(arguments));
  if (status != cudaSuccess) {
    return status;
  }
  status = cutlass_status_to_cuda(gemm.initialize(arguments, workspace));
  if (status != cudaSuccess) {
    return status;
  }
  return cutlass_status_to_cuda(gemm.run(stream));
}

template <int MmaSM, typename DTypeA, typename DTypeB, typename DTypeOut>
cudaError_t launch_cutlass_dense_gemm_sm100(void* workspace, size_t workspace_size_in_bytes,
                                            DTypeA* a_ptr, DTypeB* b_ptr, DTypeOut* d_ptr,
                                            int m, int n, int k, cudaStream_t stream) {
  // GEMM2 path: materialize both operands and use a regular tensor-op GEMM.
  using namespace cute;
  using ElementA = DTypeA;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = DTypeB;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = DTypeOut;
  using LayoutD = cutlass::layout::RowMajor;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void;
  using LayoutC = void;
  constexpr int AlignmentC = 0;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using MmaTileShape = Shape<Int<MmaSM * 128>, _128, _64>;
  using ClusterShape = Shape<Int<MmaSM>, _1, _1>;
  using MainloopSchedule =
      std::conditional_t<MmaSM == 2, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                         cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {a_ptr, stride_a, b_ptr, stride_b},
      {{}, d_ptr, stride_d, d_ptr, stride_d}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  Gemm gemm;
  size_t required_workspace = Gemm::get_workspace_size(arguments);
  if (required_workspace > workspace_size_in_bytes) {
    return cudaErrorMemoryAllocation;
  }

  cudaError_t status = cutlass_status_to_cuda(gemm.can_implement(arguments));
  if (status != cudaSuccess) {
    return status;
  }
  status = cutlass_status_to_cuda(gemm.initialize(arguments, workspace));
  if (status != cudaSuccess) {
    return status;
  }
  return cutlass_status_to_cuda(gemm.run(stream));
}

#endif

// Temporary FP8-unit emulation: quantize intermediate values to an E4M3FN-like grid.
__device__ __forceinline__ float quantize_e4m3fn_like(float x) {
  if (!isfinite(x) || x == 0.0f) return 0.0f;
  float sign = x < 0.0f ? -1.0f : 1.0f;
  float ax = fabsf(x);
  const float kMax = 448.0f;
  const float kMinSub = ldexpf(1.0f, -9);  // 2^-9
  if (ax >= kMax) return sign * kMax;
  if (ax < kMinSub) return 0.0f;

  int e2;
  float m = frexpf(ax, &e2);  // ax = m * 2^e2, m in [0.5,1)
  int e = e2 - 1;             // normalized exponent in [-6, 8]

  if (e < -6) {
    float q = nearbyintf(ax / kMinSub);
    q = fminf(7.0f, fmaxf(0.0f, q));
    return sign * q * kMinSub;
  }
  if (e > 8) return sign * kMax;

  float base = ldexpf(1.0f, e);
  float mf = ax / base;                         // [1,2)
  float qm = nearbyintf((mf - 1.0f) * 8.0f);   // 3-bit frac
  qm = fminf(7.0f, fmaxf(0.0f, qm));
  float qmf = 1.0f + qm * 0.125f;
  return sign * qmf * base;
}

__global__ void gemm1_kernel(const float* __restrict__ a, int64_t t, int hidden, int gemm1_out,
                             int block, int hidden_blocks, int local_expert_idx,
                             const float* __restrict__ local_weight, const uint8_t* __restrict__ w13,
                             const float* __restrict__ s13, bool emulate_fp8_unit,
                             bool emulate_fp16_operands, bool emulate_acc_half,
                             float* __restrict__ g1) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = t * gemm1_out;
  if (idx >= total) return;

  int64_t tok = idx / gemm1_out;
  int j = static_cast<int>(idx - tok * gemm1_out);

  float w_tok = local_weight[tok * 32 + local_expert_idx];
  if (w_tok == 0.0f) {
    g1[idx] = 0.0f;
    return;
  }

  int jb = j / block;
  const float* a_row = a + tok * hidden;
  const uint8_t* w_row = w13 + static_cast<int64_t>(j) * hidden;
  float acc = 0.0f;
  __half acc_h = __float2half(0.0f);
  for (int hb = 0; hb < hidden_blocks; ++hb) {
    float scale = s13[jb * hidden_blocks + hb];
    float block_raw = 0.0f;
    int h0 = hb * block;
    for (int u = 0; u < block; ++u) {
      int h = h0 + u;
      float wv_raw = fp8_e4m3fn_to_float_device(w_row[h]);
      float av = a_row[h];
      if (emulate_fp8_unit) {
        av = quantize_e4m3fn_like(av);
        wv_raw = quantize_e4m3fn_like(wv_raw);
      }
      float prod_raw;
      if (emulate_fp16_operands) {
        __half av_h = __float2half(av);
        __half wv_h = __float2half(wv_raw);
        prod_raw = __half2float(__hmul(av_h, wv_h));
      } else {
        prod_raw = av * wv_raw;
      }
      block_raw += prod_raw;
    }
    float block_val = block_raw * scale;
    if (emulate_acc_half) {
      // Optional narrow-accumulator mode.
      acc_h = __hadd(acc_h, __float2half(block_val));
    } else {
      acc += block_val;
    }
  }
  g1[idx] = emulate_acc_half ? __half2float(acc_h) : acc;
}

__global__ void swiglu_kernel(const float* __restrict__ g1, int64_t t, int intermediate,
                              int local_expert_idx, const float* __restrict__ local_weight,
                              bool emulate_fp8_unit, float* __restrict__ c) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = t * intermediate;
  if (idx >= total) return;
  int64_t tok = idx / intermediate;
  int i = static_cast<int>(idx - tok * intermediate);

  float w_tok = local_weight[tok * 32 + local_expert_idx];
  if (w_tok == 0.0f) {
    c[idx] = 0.0f;
    return;
  }

  const float* g1_row = g1 + tok * (2 * intermediate);
  float x1 = g1_row[i];
  float x2 = g1_row[i + intermediate];
  float y = x1 * siluf_device(x2);
  // Keep FP32 activation path in TC-like emulation mode.
  (void)emulate_fp8_unit;
  c[idx] = y;
}

// Permuted-path GEMM1. `a` is the original [T, H] activation tensor; we gather
// rows via `permuted_tok[pr]` rather than expanding into a compact buffer, to
// save a DRAM pass. Output is compact: `g1_perm[n_rows, gemm1_out]`.
__global__ void gemm1_permuted_kernel(const float* __restrict__ a, int hidden, int gemm1_out,
                                      int block, int hidden_blocks, int n_rows,
                                      const int* __restrict__ permuted_tok,
                                      const uint8_t* __restrict__ w13,
                                      const float* __restrict__ s13, bool emulate_fp8_unit,
                                      bool emulate_fp16_operands, bool emulate_acc_half,
                                      float* __restrict__ g1_perm) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * gemm1_out;
  if (idx >= total) return;

  int pr = static_cast<int>(idx / gemm1_out);
  int j = static_cast<int>(idx - static_cast<int64_t>(pr) * gemm1_out);
  int tok = permuted_tok[pr];

  int jb = j / block;
  const float* a_row = a + static_cast<int64_t>(tok) * hidden;
  const uint8_t* w_row = w13 + static_cast<int64_t>(j) * hidden;
  float acc = 0.0f;
  __half acc_h = __float2half(0.0f);
  for (int hb = 0; hb < hidden_blocks; ++hb) {
    float scale = s13[jb * hidden_blocks + hb];
    float block_raw = 0.0f;
    int h0 = hb * block;
    for (int u = 0; u < block; ++u) {
      int h = h0 + u;
      float wv_raw = fp8_e4m3fn_to_float_device(w_row[h]);
      float av = a_row[h];
      if (emulate_fp8_unit) {
        av = quantize_e4m3fn_like(av);
        wv_raw = quantize_e4m3fn_like(wv_raw);
      }
      float prod_raw;
      if (emulate_fp16_operands) {
        __half av_h = __float2half(av);
        __half wv_h = __float2half(wv_raw);
        prod_raw = __half2float(__hmul(av_h, wv_h));
      } else {
        prod_raw = av * wv_raw;
      }
      block_raw += prod_raw;
    }
    float block_val = block_raw * scale;
    if (emulate_acc_half) {
      acc_h = __hadd(acc_h, __float2half(block_val));
    } else {
      acc += block_val;
    }
  }
  g1_perm[idx] = emulate_acc_half ? __half2float(acc_h) : acc;
}

__global__ void gemm1_compact_kernel(const float* __restrict__ a_compact, int hidden,
                                     int gemm1_out, int block, int hidden_blocks, int n_rows,
                                     const uint8_t* __restrict__ w13,
                                     const float* __restrict__ s13, bool emulate_fp8_unit,
                                     bool emulate_fp16_operands, bool emulate_acc_half,
                                     float* __restrict__ g1_out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * gemm1_out;
  if (idx >= total) return;

  int row = static_cast<int>(idx / gemm1_out);
  int j = static_cast<int>(idx - static_cast<int64_t>(row) * gemm1_out);
  int jb = j / block;

  const float* a_row = a_compact + static_cast<int64_t>(row) * hidden;
  const uint8_t* w_row = w13 + static_cast<int64_t>(j) * hidden;
  float acc = 0.0f;
  __half acc_h = __float2half(0.0f);
  for (int hb = 0; hb < hidden_blocks; ++hb) {
    float scale = s13[jb * hidden_blocks + hb];
    float block_raw = 0.0f;
    int h0 = hb * block;
    for (int u = 0; u < block; ++u) {
      int h = h0 + u;
      float wv_raw = fp8_e4m3fn_to_float_device(w_row[h]);
      float av = a_row[h];
      if (emulate_fp8_unit) {
        av = quantize_e4m3fn_like(av);
        wv_raw = quantize_e4m3fn_like(wv_raw);
      }
      float prod_raw;
      if (emulate_fp16_operands) {
        __half av_h = __float2half(av);
        __half wv_h = __float2half(wv_raw);
        prod_raw = __half2float(__hmul(av_h, wv_h));
      } else {
        prod_raw = av * wv_raw;
      }
      block_raw += prod_raw;
    }
    float block_val = block_raw * scale;
    if (emulate_acc_half) {
      acc_h = __hadd(acc_h, __float2half(block_val));
    } else {
      acc += block_val;
    }
  }
  g1_out[idx] = emulate_acc_half ? __half2float(acc_h) : acc;
}

// Permuted swiglu: input/output are both compact n_rows-indexed, no masking.
__global__ void swiglu_permuted_kernel(const float* __restrict__ g1_perm, int intermediate,
                                       int n_rows, bool emulate_fp8_unit,
                                       float* __restrict__ c_perm) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * intermediate;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / intermediate);
  int i = static_cast<int>(idx - static_cast<int64_t>(pr) * intermediate);
  const float* g1_row = g1_perm + static_cast<int64_t>(pr) * (2 * intermediate);
  float x1 = g1_row[i];
  float x2 = g1_row[i + intermediate];
  (void)emulate_fp8_unit;  // Activation path stays FP32 in TC-like emulation.
  c_perm[idx] = x1 * siluf_device(x2);
}

// Permuted GEMM2 + scatter-accumulate into the global [T, H] out_acc tensor.
// Within a single launch each (pr, h) touches a unique out_acc cell because a
// token routes to any given expert at most once; across expert launches the
// stream orders contributions, so no atomicAdd is needed.
__global__ void gemm2_scatter_accumulate_kernel(const float* __restrict__ c_perm, int hidden,
                                                int intermediate, int block,
                                                int intermediate_blocks, int n_rows,
                                                const int* __restrict__ permuted_tok,
                                                const float* __restrict__ permuted_w,
                                                const uint8_t* __restrict__ w2,
                                                const float* __restrict__ s2,
                                                bool emulate_fp8_unit,
                                                bool emulate_fp16_operands, bool emulate_acc_half,
                                                float* __restrict__ out_acc) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * hidden;
  if (idx >= total) return;

  int pr = static_cast<int>(idx / hidden);
  int h = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden);
  int tok = permuted_tok[pr];
  float w_tok = permuted_w[pr];

  int hb = h / block;
  const float* c_row = c_perm + static_cast<int64_t>(pr) * intermediate;
  const uint8_t* w_row = w2 + static_cast<int64_t>(h) * intermediate;
  float acc = 0.0f;
  __half acc_h = __float2half(0.0f);
  for (int ib = 0; ib < intermediate_blocks; ++ib) {
    float scale = s2[hb * intermediate_blocks + ib];
    float block_raw = 0.0f;
    int i0 = ib * block;
    for (int u = 0; u < block; ++u) {
      int i = i0 + u;
      float wv_raw = fp8_e4m3fn_to_float_device(w_row[i]);
      float cv = c_row[i];
      if (emulate_fp8_unit) {
        cv = quantize_e4m3fn_like(cv);
        wv_raw = quantize_e4m3fn_like(wv_raw);
      }
      float prod_raw;
      if (emulate_fp16_operands) {
        __half cv_h = __float2half(cv);
        __half wv_h = __float2half(wv_raw);
        prod_raw = __half2float(__hmul(cv_h, wv_h));
      } else {
        prod_raw = cv * wv_raw;
      }
      block_raw += prod_raw;
    }
    float block_val = block_raw * scale;
    if (emulate_acc_half) {
      acc_h = __hadd(acc_h, __float2half(block_val));
    } else {
      acc += block_val;
    }
  }
  acc = emulate_acc_half ? __half2float(acc_h) : acc;
  out_acc[static_cast<int64_t>(tok) * hidden + h] += w_tok * acc;
}

__global__ void gemm2_acc_kernel(const float* __restrict__ c, int64_t t, int hidden, int intermediate,
                                 int block, int intermediate_blocks, int local_expert_idx,
                                 const float* __restrict__ local_weight, const uint8_t* __restrict__ w2,
                                 const float* __restrict__ s2, bool emulate_fp8_unit,
                                 bool emulate_fp16_operands, bool emulate_acc_half,
                                 float* __restrict__ out_acc) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = t * hidden;
  if (idx >= total) return;

  int64_t tok = idx / hidden;
  int h = static_cast<int>(idx - tok * hidden);

  float w_tok = local_weight[tok * 32 + local_expert_idx];
  if (w_tok == 0.0f) return;

  int hb = h / block;
  const float* c_row = c + tok * intermediate;
  const uint8_t* w_row = w2 + static_cast<int64_t>(h) * intermediate;
  float acc = 0.0f;
  __half acc_h = __float2half(0.0f);
  for (int ib = 0; ib < intermediate_blocks; ++ib) {
    float scale = s2[hb * intermediate_blocks + ib];
    float block_raw = 0.0f;
    int i0 = ib * block;
    for (int u = 0; u < block; ++u) {
      int i = i0 + u;
      float wv_raw = fp8_e4m3fn_to_float_device(w_row[i]);
      float cv = c_row[i];
      if (emulate_fp8_unit) {
        cv = quantize_e4m3fn_like(cv);
        wv_raw = quantize_e4m3fn_like(wv_raw);
      }
      float prod_raw;
      if (emulate_fp16_operands) {
        __half cv_h = __float2half(cv);
        __half wv_h = __float2half(wv_raw);
        prod_raw = __half2float(__hmul(cv_h, wv_h));
      } else {
        prod_raw = cv * wv_raw;
      }
      block_raw += prod_raw;
    }
    float block_val = block_raw * scale;
    if (emulate_acc_half) {
      // Optional narrow-accumulator mode.
      acc_h = __hadd(acc_h, __float2half(block_val));
    } else {
      acc += block_val;
    }
  }

  acc = emulate_acc_half ? __half2float(acc_h) : acc;
  out_acc[idx] += w_tok * acc;
}

__global__ void write_single_group_indptr_kernel(int padded_rows, int* __restrict__ indptr) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    indptr[0] = 0;
    indptr[1] = padded_rows;
  }
}

__global__ void gather_hidden_fp8_rows_kernel(const uint8_t* __restrict__ hidden_fp8,
                                             const int* __restrict__ permuted_tok,
                                             int n_rows, int padded_rows, int hidden,
                                             uint8_t* __restrict__ a_perm) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(padded_rows) * hidden;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / hidden);
  int h = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden);
  if (pr >= n_rows) {
    a_perm[idx] = 0;
    return;
  }
  int tok = permuted_tok[pr];
  a_perm[idx] = hidden_fp8[static_cast<int64_t>(tok) * hidden + h];
}

__global__ void gather_hidden_scale_rows_kernel(const float* __restrict__ hidden_scale,
                                               const int* __restrict__ permuted_tok,
                                               int64_t t, int n_rows, int padded_rows,
                                               int hidden_blocks, float* __restrict__ a_scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(padded_rows) * hidden_blocks;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / hidden_blocks);
  int hb = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden_blocks);
  if (pr >= n_rows) {
    a_scale[static_cast<int64_t>(hb) * padded_rows + pr] = 1.0f;
    return;
  }
  int tok = permuted_tok[pr];
  a_scale[static_cast<int64_t>(hb) * padded_rows + pr] =
      hidden_scale[static_cast<int64_t>(hb) * t + tok];
}

__global__ void gather_hidden_scale_rows_mn_major_kernel(const float* __restrict__ hidden_scale,
                                                         const int* __restrict__ permuted_tok,
                                                         int64_t t, int n_rows, int padded_rows,
                                                         int hidden_blocks,
                                                         float* __restrict__ a_scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(padded_rows) * hidden_blocks;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / hidden_blocks);
  int hb = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden_blocks);
  if (pr >= n_rows) {
    a_scale[static_cast<int64_t>(pr) * hidden_blocks + hb] = 1.0f;
    return;
  }
  int tok = permuted_tok[pr];
  a_scale[static_cast<int64_t>(pr) * hidden_blocks + hb] =
      hidden_scale[static_cast<int64_t>(hb) * t + tok];
}

__global__ void transpose_rowmajor_nk_to_colmajor_nk_kernel(const uint8_t* __restrict__ src,
                                                           int n, int k,
                                                           uint8_t* __restrict__ dst) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n) * k;
  if (idx >= total) return;
  int row_n = static_cast<int>(idx / k);
  int col_k = static_cast<int>(idx - static_cast<int64_t>(row_n) * k);
  dst[static_cast<int64_t>(col_k) * n + row_n] = src[idx];
}

__global__ void transpose_scale_nblock_kblock_to_kblock_nblock_kernel(
    const float* __restrict__ src, int n_blocks, int k_blocks, float* __restrict__ dst) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_blocks * k_blocks;
  if (idx >= total) return;
  int nb = idx / k_blocks;
  int kb = idx - nb * k_blocks;
  dst[kb * n_blocks + nb] = src[idx];
}

__global__ void copy_scale_nblock_kblock_kernel(const float* __restrict__ src, int total,
                                                float* __restrict__ dst) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  dst[idx] = src[idx];
}

__global__ void dequant_fp8_rows_kernel(const uint8_t* __restrict__ in_fp8,
                                        const float* __restrict__ in_scale, int rows, int cols,
                                        int row_scale_granularity, int col_blocks,
                                        bool scale_major_k,
                                        float* __restrict__ out_f32) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(rows) * cols;
  if (idx >= total) return;
  int row = static_cast<int>(idx / cols);
  int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  int rb = row / row_scale_granularity;
  int cb = col / 128;
  int row_scale_count = (rows + row_scale_granularity - 1) / row_scale_granularity;
  float scale = scale_major_k ? in_scale[static_cast<int64_t>(rb) * col_blocks + cb]
                              : in_scale[static_cast<int64_t>(cb) * row_scale_count + rb];
  out_f32[idx] = fp8_e4m3fn_to_float_device(in_fp8[idx]) * scale;
}

__global__ void dequant_fp8_rows_native_kernel(const uint8_t* __restrict__ in_fp8,
                                               const float* __restrict__ in_scale, int rows,
                                               int cols, int row_scale_granularity,
                                               int col_blocks, bool scale_major_k,
                                               float* __restrict__ out_f32) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(rows) * cols;
  if (idx >= total) return;
  int row = static_cast<int>(idx / cols);
  int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  int rb = row / row_scale_granularity;
  int cb = col / 128;
  int row_scale_count = (rows + row_scale_granularity - 1) / row_scale_granularity;
  float scale = scale_major_k ? in_scale[static_cast<int64_t>(rb) * col_blocks + cb]
                              : in_scale[static_cast<int64_t>(cb) * row_scale_count + rb];
  out_f32[idx] = fp8_native_to_float_device(in_fp8[idx]) * scale;
}

__global__ void dequant_fp8_rows_native_to_f16_kernel(const uint8_t* __restrict__ in_fp8,
                                                      const float* __restrict__ in_scale,
                                                      int rows, int cols,
                                                      int row_scale_granularity,
                                                      int col_blocks, bool scale_major_k,
                                                      uint16_t* __restrict__ out_f16) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(rows) * cols;
  if (idx >= total) return;
  int row = static_cast<int>(idx / cols);
  int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  int rb = row / row_scale_granularity;
  int cb = col / 128;
  int row_scale_count = (rows + row_scale_granularity - 1) / row_scale_granularity;
  float scale = scale_major_k ? in_scale[static_cast<int64_t>(rb) * col_blocks + cb]
                              : in_scale[static_cast<int64_t>(cb) * row_scale_count + rb];
  union {
    uint16_t u;
    __half h;
  } v;
  v.h = __float2half(fp8_e4m3fn_to_float_device(in_fp8[idx]) * scale);
  out_f16[idx] = v.u;
}

__global__ void dequant_fp8_rows_native_to_bf16_kernel(const uint8_t* __restrict__ in_fp8,
                                                       const float* __restrict__ in_scale,
                                                       int rows, int cols,
                                                       int row_scale_granularity,
                                                       int col_blocks, bool scale_major_k,
                                                       uint16_t* __restrict__ out_bf16) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(rows) * cols;
  if (idx >= total) return;
  int row = static_cast<int>(idx / cols);
  int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  int rb = row / row_scale_granularity;
  int cb = col / 128;
  int row_scale_count = (rows + row_scale_granularity - 1) / row_scale_granularity;
  float scale = scale_major_k ? in_scale[static_cast<int64_t>(rb) * col_blocks + cb]
                              : in_scale[static_cast<int64_t>(cb) * row_scale_count + rb];
  union {
    __nv_bfloat16 bf16;
    uint16_t u;
  } v;
  v.bf16 = __float2bfloat16_rn(fp8_e4m3fn_to_float_device(in_fp8[idx]) * scale);
  out_bf16[idx] = v.u;
}

__global__ void swiglu_quantize_bf16_to_fp8_kernel(const uint16_t* __restrict__ g1_bf16,
                                                   int intermediate, int n_rows, int padded_rows,
                                                   uint8_t* __restrict__ c_fp8,
                                                   float* __restrict__ c_scale,
                                                   bool scale_major_k) {
  int row = blockIdx.x;
  int ib = blockIdx.y;
  int tid = threadIdx.x;
  if (row >= padded_rows) return;

  extern __shared__ float smem[];
  float v = 0.0f;
  int i = ib * 128 + tid;
  if (tid < 128 && i < intermediate && row < n_rows) {
    const uint16_t* g1_row = g1_bf16 + static_cast<int64_t>(row) * (2 * intermediate);
    float x1 = bf16_to_float_device(g1_row[i]);
    float x2 = bf16_to_float_device(g1_row[i + intermediate]);
    v = x1 * siluf_device(x2);
  }
  smem[tid] = fabsf(v);
  __syncthreads();

  for (int offset = 64; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    __syncthreads();
  }

  float scale = fmaxf(smem[0] / 448.0f, 1.0e-8f);
  if (tid == 0) {
    if (scale_major_k) {
      c_scale[static_cast<int64_t>(row) * (intermediate / 128) + ib] = scale;
    } else {
      c_scale[static_cast<int64_t>(ib) * padded_rows + row] = scale;
    }
  }
  if (tid < 128 && i < intermediate) {
    c_fp8[static_cast<int64_t>(row) * intermediate + i] =
        (row < n_rows) ? float_to_e4m3_device(v / scale) : 0;
  }
}

__global__ void swiglu_quantize_f16_to_fp8_kernel(const uint16_t* __restrict__ g1_f16,
                                                  int intermediate, int n_rows, int padded_rows,
                                                  uint8_t* __restrict__ c_fp8,
                                                  float* __restrict__ c_scale,
                                                  bool scale_major_k) {
  int row = blockIdx.x;
  int ib = blockIdx.y;
  int tid = threadIdx.x;
  if (row >= padded_rows) return;

  extern __shared__ float smem[];
  float v = 0.0f;
  int i = ib * 128 + tid;
  if (tid < 128 && i < intermediate && row < n_rows) {
    const uint16_t* g1_row = g1_f16 + static_cast<int64_t>(row) * (2 * intermediate);
    float x1 = f16_to_float_device(g1_row[i]);
    float x2 = f16_to_float_device(g1_row[i + intermediate]);
    v = x1 * siluf_device(x2);
  }
  smem[tid] = fabsf(v);
  __syncthreads();

  for (int offset = 64; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    __syncthreads();
  }

  float scale = fmaxf(smem[0] / 448.0f, 1.0e-8f);
  if (tid == 0) {
    if (scale_major_k) {
      c_scale[static_cast<int64_t>(row) * (intermediate / 128) + ib] = scale;
    } else {
      c_scale[static_cast<int64_t>(ib) * padded_rows + row] = scale;
    }
  }
  if (tid < 128 && i < intermediate) {
    c_fp8[static_cast<int64_t>(row) * intermediate + i] =
        (row < n_rows) ? float_to_e4m3_device(v / scale) : 0;
  }
}

__global__ void quantize_float_rows_to_fp8_kernel(const float* __restrict__ in_f32, int cols,
                                                  int n_rows, int padded_rows,
                                                  uint8_t* __restrict__ out_fp8,
                                                  float* __restrict__ out_scale,
                                                  bool scale_major_k) {
  int row = blockIdx.x;
  int cb = blockIdx.y;
  int tid = threadIdx.x;
  if (row >= padded_rows) return;

  extern __shared__ float smem[];
  float v = 0.0f;
  int col = cb * 128 + tid;
  if (tid < 128 && col < cols && row < n_rows) {
    v = in_f32[static_cast<int64_t>(row) * cols + col];
  }
  smem[tid] = fabsf(v);
  __syncthreads();

  for (int offset = 64; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    __syncthreads();
  }

  float scale = fmaxf(smem[0] / 448.0f, 1.0e-8f);
  if (tid == 0) {
    int col_blocks = cols / 128;
    if (scale_major_k) {
      out_scale[static_cast<int64_t>(row) * col_blocks + cb] = scale;
    } else {
      out_scale[static_cast<int64_t>(cb) * padded_rows + row] = scale;
    }
  }
  if (tid < 128 && col < cols) {
    out_fp8[static_cast<int64_t>(row) * cols + col] =
        (row < n_rows) ? float_to_e4m3_device(v / scale) : 0;
  }
}

__global__ void quantize_float_rows_to_fp8_mse_kernel(const float* __restrict__ in_f32, int cols,
                                                      int n_rows, int padded_rows,
                                                      uint8_t* __restrict__ out_fp8,
                                                      float* __restrict__ out_scale,
                                                      bool scale_major_k) {
  int row = blockIdx.x;
  int cb = blockIdx.y;
  int tid = threadIdx.x;
  if (row >= padded_rows) return;

  __shared__ float vals[128];
  __shared__ float best_scale;
  int col = cb * 128 + tid;
  float v = 0.0f;
  if (tid < 128 && col < cols && row < n_rows) {
    v = in_f32[static_cast<int64_t>(row) * cols + col];
  }
  vals[tid] = v;
  __syncthreads();

  if (tid == 0) {
    float absmax = 0.0f;
    for (int i = 0; i < 128; ++i) {
      float av = fabsf(vals[i]);
      absmax = fmaxf(absmax, av);
    }

    float base_scale = fmaxf(absmax / 448.0f, 1.0e-8f);
    float chosen_scale = base_scale;
    float best_err = INFINITY;

    // Search a small logarithmic window around absmax scaling and keep the
    // scale that minimizes actual FP8 encode/decode MSE for this 1x128 block.
    for (int step = -12; step <= 12; ++step) {
      float cand_scale = base_scale * exp2f(static_cast<float>(step) * 0.25f);
      cand_scale = fmaxf(cand_scale, 1.0e-8f);
      float err = 0.0f;
      for (int i = 0; i < 128; ++i) {
        uint8_t q = float_to_e4m3_device(vals[i] / cand_scale);
        float dq = fp8_native_to_float_device(q) * cand_scale;
        float diff = vals[i] - dq;
        err += diff * diff;
      }
      if (err < best_err) {
        best_err = err;
        chosen_scale = cand_scale;
      }
    }
    best_scale = chosen_scale;
    int col_blocks = cols / 128;
    if (scale_major_k) {
      out_scale[static_cast<int64_t>(row) * col_blocks + cb] = chosen_scale;
    } else {
      out_scale[static_cast<int64_t>(cb) * padded_rows + row] = chosen_scale;
    }
  }
  __syncthreads();

  if (tid < 128 && col < cols) {
    out_fp8[static_cast<int64_t>(row) * cols + col] =
        (row < n_rows) ? float_to_e4m3_device(vals[tid] / best_scale) : 0;
  }
}

__global__ void compute_block_scale_128x128_kernel(const float* __restrict__ in_f32, int cols,
                                                   int n_rows, int padded_rows,
                                                   int row_blocks, int col_blocks,
                                                   float* __restrict__ out_scale,
                                                   bool scale_major_k) {
  int rb = blockIdx.x;
  int cb = blockIdx.y;
  int tid = threadIdx.x;
  if (rb >= row_blocks || cb >= col_blocks) return;

  extern __shared__ float smem[];
  float thread_max = 0.0f;
  int row0 = rb * 128;
  int col0 = cb * 128;
  int elems = 128 * 128;
  for (int linear = tid; linear < elems; linear += blockDim.x) {
    int r = row0 + linear / 128;
    int c = col0 + linear % 128;
    float v = 0.0f;
    if (r < n_rows && c < cols) {
      v = in_f32[static_cast<int64_t>(r) * cols + c];
    }
    thread_max = fmaxf(thread_max, fabsf(v));
  }
  smem[tid] = thread_max;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    __syncthreads();
  }
  if (tid == 0) {
    float scale = fmaxf(smem[0] / 448.0f, 1.0e-8f);
    if (scale_major_k) {
      out_scale[static_cast<int64_t>(rb) * col_blocks + cb] = scale;
    } else {
      out_scale[static_cast<int64_t>(cb) * row_blocks + rb] = scale;
    }
  }
}

__global__ void quantize_float_blocks_128x128_to_fp8_kernel(const float* __restrict__ in_f32,
                                                            const float* __restrict__ in_scale,
                                                            int cols, int n_rows,
                                                            int padded_rows, int row_blocks,
                                                            int col_blocks,
                                                            uint8_t* __restrict__ out_fp8,
                                                            bool scale_major_k) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(padded_rows) * cols;
  if (idx >= total) return;
  int row = static_cast<int>(idx / cols);
  int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  int rb = row / 128;
  int cb = col / 128;
  float scale = scale_major_k ? in_scale[static_cast<int64_t>(rb) * col_blocks + cb]
                              : in_scale[static_cast<int64_t>(cb) * row_blocks + rb];
  float v = (row < n_rows) ? in_f32[idx] : 0.0f;
  out_fp8[idx] = (row < n_rows) ? float_to_e4m3_device(v / scale) : 0;
}

__global__ void scatter_float_weighted_kernel(const float* __restrict__ d_f32, int hidden,
                                              int n_rows, const int* __restrict__ permuted_tok,
                                              const float* __restrict__ permuted_w,
                                              float* __restrict__ out_acc) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * hidden;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / hidden);
  int h = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden);
  int tok = permuted_tok[pr];
  float w = permuted_w[pr];
  out_acc[static_cast<int64_t>(tok) * hidden + h] += w * d_f32[idx];
}

__global__ void scatter_float_weighted_row_scaled_kernel(
    const float* __restrict__ d_f32, const float* __restrict__ row_scale, int hidden, int n_rows,
    const int* __restrict__ permuted_tok, const float* __restrict__ permuted_w,
    float* __restrict__ out_acc) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(n_rows) * hidden;
  if (idx >= total) return;
  int pr = static_cast<int>(idx / hidden);
  int h = static_cast<int>(idx - static_cast<int64_t>(pr) * hidden);
  int tok = permuted_tok[pr];
  float w = permuted_w[pr] * row_scale[pr];
  out_acc[static_cast<int64_t>(tok) * hidden + h] += w * d_f32[idx];
}

__global__ void bf16_matrix_to_float_kernel(const uint16_t* __restrict__ in, int64_t n,
                                           float* __restrict__ out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = bf16_to_float_device(in[idx]);
}

__global__ void f16_matrix_to_float_kernel(const uint16_t* __restrict__ in, int64_t n,
                                          float* __restrict__ out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = f16_to_float_device(in[idx]);
}

__global__ void float_to_f16_kernel(const float* __restrict__ in, int64_t n,
                                    uint16_t* __restrict__ out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  union {
    uint16_t u;
    __half h;
  } v;
  v.h = __float2half(in[idx]);
  out[idx] = v.u;
}

__global__ void float_rows_scaled_to_f16_kernel(const float* __restrict__ in, int cols,
                                                int n_rows, int padded_rows,
                                                uint16_t* __restrict__ out,
                                                float* __restrict__ row_scale) {
  int row = blockIdx.x;
  if (row >= padded_rows) return;
  __shared__ float smem[256];
  float max_abs = 0.0f;
  if (row < n_rows) {
    const float* in_row = in + static_cast<int64_t>(row) * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
      max_abs = fmaxf(max_abs, fabsf(in_row[col]));
    }
  }
  smem[threadIdx.x] = max_abs;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float scale = fmaxf(smem[0] / 60000.0f, 1.0f);
  if (threadIdx.x == 0) row_scale[row] = scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float v = (row < n_rows) ? in[static_cast<int64_t>(row) * cols + col] / scale : 0.0f;
    union {
      __half h;
      uint16_t u;
    } packed;
    packed.h = __float2half(v);
    out[static_cast<int64_t>(row) * cols + col] = packed.u;
  }
}

__global__ void float_to_bf16_kernel(const float* __restrict__ in, int64_t n,
                                     uint16_t* __restrict__ out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  union {
    __nv_bfloat16 bf16;
    uint16_t u;
  } v;
  v.bf16 = __float2bfloat16_rn(in[idx]);
  out[idx] = v.u;
}

__global__ void compare_abs_diff_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                        int64_t n, float* __restrict__ max_out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float diff = fabsf(a[idx] - b[idx]);
  atomicMax(reinterpret_cast<int*>(max_out), __float_as_int(diff));
}

}  // namespace

float fp8_e4m3fn_to_float(uint8_t x) {
  int sign = (x & 0x80) ? -1 : 1;
  int exp = (x >> 3) & 0x0f;
  int mant = x & 0x07;

  if (exp == 0) {
    if (mant == 0) {
      return sign == 1 ? 0.0f : -0.0f;
    }
    float frac = static_cast<float>(mant) / 8.0f;
    return sign * std::ldexp(frac, -6);
  }

  float frac = 1.0f + static_cast<float>(mant) / 8.0f;
  return sign * std::ldexp(frac, exp - 7);
}

HostMxfpGemmModule::HostMxfpGemmModule(int hidden, int intermediate, int block)
    : hidden_(hidden),
      intermediate_(intermediate),
      block_(block),
      gemm1_out_(2 * intermediate),
      hidden_blocks_(hidden / block),
      intermediate_blocks_(intermediate / block),
      gemm1_out_blocks_(gemm1_out_ / block),
      w13_fp8_(gemm1_weight_elems()),
      w13_scale_(gemm1_scale_elems()),
      w2_fp8_(gemm2_weight_elems()),
      w2_scale_(gemm2_scale_elems()) {}

size_t HostMxfpGemmModule::gemm1_weight_elems() const {
  return static_cast<size_t>(gemm1_out_) * static_cast<size_t>(hidden_);
}

size_t HostMxfpGemmModule::gemm1_scale_elems() const {
  return static_cast<size_t>(gemm1_out_blocks_) * static_cast<size_t>(hidden_blocks_);
}

size_t HostMxfpGemmModule::gemm2_weight_elems() const {
  return static_cast<size_t>(hidden_) * static_cast<size_t>(intermediate_);
}

size_t HostMxfpGemmModule::gemm2_scale_elems() const {
  return static_cast<size_t>(hidden_blocks_) * static_cast<size_t>(intermediate_blocks_);
}

void HostMxfpGemmModule::load_expert_from_device(int local_expert_idx, const uint8_t* gemm1_weights_dev,
                                                  const float* gemm1_scales_dev,
                                                  const uint8_t* gemm2_weights_dev,
                                                  const float* gemm2_scales_dev,
                                                  cudaStream_t stream) {
  size_t w13_elems = gemm1_weight_elems();
  size_t w13s_elems = gemm1_scale_elems();
  size_t w2_elems = gemm2_weight_elems();
  size_t w2s_elems = gemm2_scale_elems();

  size_t w13_off = static_cast<size_t>(local_expert_idx) * w13_elems;
  size_t w13s_off = static_cast<size_t>(local_expert_idx) * w13s_elems;
  size_t w2_off = static_cast<size_t>(local_expert_idx) * w2_elems;
  size_t w2s_off = static_cast<size_t>(local_expert_idx) * w2s_elems;

  cudaMemcpyAsync(w13_fp8_.data(), gemm1_weights_dev + w13_off, w13_elems, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(w13_scale_.data(), gemm1_scales_dev + w13s_off, w13s_elems * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(w2_fp8_.data(), gemm2_weights_dev + w2_off, w2_elems, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(w2_scale_.data(), gemm2_scales_dev + w2s_off, w2s_elems * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
}

void HostMxfpGemmModule::gemm1_matvec(const float* a_row, float* g1_out) const {
  for (int j = 0; j < gemm1_out_; ++j) {
    int jb = j / block_;
    float acc = 0.0f;
    const uint8_t* w13_row = w13_fp8_.data() + static_cast<size_t>(j) * hidden_;
    for (int h = 0; h < hidden_; ++h) {
      int hb = h / block_;
      float s = w13_scale_[static_cast<size_t>(jb) * hidden_blocks_ + static_cast<size_t>(hb)];
      float wv = fp8_e4m3fn_to_float(w13_row[h]) * s;
      acc += a_row[h] * wv;
    }
    g1_out[j] = acc;
  }
}

void HostMxfpGemmModule::swiglu(const float* g1, int intermediate, float* c_out) {
  for (int i = 0; i < intermediate; ++i) {
    float x1 = g1[i];
    float x2 = g1[i + intermediate];
    c_out[i] = x1 * siluf_host(x2);
  }
}

void HostMxfpGemmModule::gemm2_matvec_accumulate(const float* c, float weight, float* out_row) const {
  for (int h = 0; h < hidden_; ++h) {
    int hb = h / block_;
    float acc = 0.0f;
    const uint8_t* w2_row = w2_fp8_.data() + static_cast<size_t>(h) * intermediate_;
    for (int i = 0; i < intermediate_; ++i) {
      int ib = i / block_;
      float s = w2_scale_[static_cast<size_t>(hb) * intermediate_blocks_ + static_cast<size_t>(ib)];
      float wv = fp8_e4m3fn_to_float(w2_row[i]) * s;
      acc += c[i] * wv;
    }
    out_row[h] += weight * acc;
  }
}

DeviceMxfpGemmModule::DeviceMxfpGemmModule(int hidden, int intermediate, int block)
    : hidden_(hidden),
      intermediate_(intermediate),
      block_(block),
      gemm1_out_(2 * intermediate),
      hidden_blocks_(hidden / block),
      intermediate_blocks_(intermediate / block),
      gemm1_out_blocks_(gemm1_out_ / block),
      max_t_(0),
      emulate_fp8_unit_(false),
      emulate_fp16_operands_(false),
      emulate_acc_half_(false),
      g1_dev_(nullptr),
      c_dev_(nullptr),
      tc_max_rows_(0),
      tc_path_enabled_(FIB_HAS_DIRECT_CUTLASS_SM100),
      tc_a_fp8_dev_(nullptr),
      tc_b_col_dev_(nullptr),
      tc_a_scale_dev_(nullptr),
      tc_b_scale_dev_(nullptr),
      tc_g1_f32_dev_(nullptr),
      tc_g1_f16_dev_(nullptr),
      tc_c_fp8_dev_(nullptr),
      tc_c_scale_dev_(nullptr),
      tc_c_bf16_dev_(nullptr),
      tc_b_bf16_dev_(nullptr),
      tc_d_f32_dev_(nullptr),
      tc_m_indptr_dev_(nullptr),
      tc_int_workspace_dev_(nullptr),
      tc_float_workspace_dev_(nullptr) {
  const char* env = std::getenv("FIB_EMULATE_FP8_UNIT");
  emulate_fp8_unit_ = (env != nullptr && env[0] == '1');
  const char* env_fp16_op = std::getenv("FIB_EMULATE_FP16_OPERANDS");
  emulate_fp16_operands_ = (env_fp16_op != nullptr && env_fp16_op[0] == '1');
  const char* env_acc = std::getenv("FIB_EMULATE_FP8_ACC_HALF");
  emulate_acc_half_ = (env_acc != nullptr && env_acc[0] == '1');
  if (emulate_fp8_unit_) {
    std::fprintf(stderr,
                 "[mxfp] FIB_EMULATE_FP8_UNIT=1 (TC-like emulation: FP8-like operands, FP32 accumulate)\n");
  }
  if (emulate_acc_half_) {
    std::fprintf(stderr, "[mxfp] FIB_EMULATE_FP8_ACC_HALF=1 (half accumulate enabled)\n");
  }
  if (emulate_fp16_operands_) {
    std::fprintf(stderr, "[mxfp] FIB_EMULATE_FP16_OPERANDS=1 (fp16*fp16 multiply emulation enabled)\n");
  }
  const char* env_tc = std::getenv("FIB_MOE_TC");
  if (env_tc != nullptr) {
    tc_path_enabled_ = (env_tc[0] != '0');
  }
  const char* env_no_tc = std::getenv("FIB_MOE_NO_TC");
  if (env_no_tc != nullptr && env_no_tc[0] == '1') {
    tc_path_enabled_ = false;
  }
  if (tc_path_enabled_ || env_tc != nullptr || env_no_tc != nullptr) {
    std::fprintf(stderr,
                 "[mxfp] CUTLASS expert GEMM path %s (%s)\n",
                 tc_path_enabled_ ? "enabled" : "disabled",
                 FIB_HAS_DIRECT_CUTLASS_SM100 ? "direct CUTLASS available"
                                              : "direct CUTLASS unavailable");
  }
  if (emulate_fp8_unit_ || emulate_fp16_operands_ || emulate_acc_half_ || env_tc != nullptr ||
      env_no_tc != nullptr || FIB_HAS_DIRECT_CUTLASS_SM100) {
    std::fflush(stderr);
  }
}

DeviceMxfpGemmModule::~DeviceMxfpGemmModule() {
  if (g1_dev_ != nullptr) cudaFree(g1_dev_);
  if (c_dev_ != nullptr) cudaFree(c_dev_);
  if (tc_a_fp8_dev_ != nullptr) cudaFree(tc_a_fp8_dev_);
  if (tc_b_col_dev_ != nullptr) cudaFree(tc_b_col_dev_);
  if (tc_a_scale_dev_ != nullptr) cudaFree(tc_a_scale_dev_);
  if (tc_b_scale_dev_ != nullptr) cudaFree(tc_b_scale_dev_);
  if (tc_g1_f32_dev_ != nullptr) cudaFree(tc_g1_f32_dev_);
  if (tc_g1_f16_dev_ != nullptr) cudaFree(tc_g1_f16_dev_);
  if (tc_c_fp8_dev_ != nullptr) cudaFree(tc_c_fp8_dev_);
  if (tc_c_scale_dev_ != nullptr) cudaFree(tc_c_scale_dev_);
  if (tc_c_bf16_dev_ != nullptr) cudaFree(tc_c_bf16_dev_);
  if (tc_b_bf16_dev_ != nullptr) cudaFree(tc_b_bf16_dev_);
  if (tc_d_f32_dev_ != nullptr) cudaFree(tc_d_f32_dev_);
  if (tc_m_indptr_dev_ != nullptr) cudaFree(tc_m_indptr_dev_);
  if (tc_int_workspace_dev_ != nullptr) cudaFree(tc_int_workspace_dev_);
  if (tc_float_workspace_dev_ != nullptr) cudaFree(tc_float_workspace_dev_);
}

void DeviceMxfpGemmModule::EnsureWorkspace(int64_t t, cudaStream_t stream) {
  (void)stream;
  if (t <= max_t_ && g1_dev_ != nullptr && c_dev_ != nullptr) return;

  if (g1_dev_ != nullptr) {
    cudaFree(g1_dev_);
    g1_dev_ = nullptr;
  }
  if (c_dev_ != nullptr) {
    cudaFree(c_dev_);
    c_dev_ = nullptr;
  }

  max_t_ = t;
  cudaError_t e1 = cudaMalloc(&g1_dev_, static_cast<size_t>(t) * static_cast<size_t>(gemm1_out_) * sizeof(float));
  cudaError_t e2 = cudaMalloc(&c_dev_, static_cast<size_t>(t) * static_cast<size_t>(intermediate_) * sizeof(float));
  if (e1 != cudaSuccess || e2 != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed for DeviceMxfpGemmModule workspace");
  }
}

bool DeviceMxfpGemmModule::SupportsTcPath() const {
  return tc_path_enabled_ && FIB_HAS_DIRECT_CUTLASS_SM100;
}

void DeviceMxfpGemmModule::EnsureTcWorkspace(int rows) {
  int padded_rows = (rows + 3) & ~3;
  if (padded_rows <= tc_max_rows_ && tc_a_fp8_dev_ != nullptr) return;

  if (tc_a_fp8_dev_ != nullptr) cudaFree(tc_a_fp8_dev_);
  if (tc_b_col_dev_ != nullptr) cudaFree(tc_b_col_dev_);
  if (tc_a_scale_dev_ != nullptr) cudaFree(tc_a_scale_dev_);
  if (tc_b_scale_dev_ != nullptr) cudaFree(tc_b_scale_dev_);
  if (tc_g1_f32_dev_ != nullptr) cudaFree(tc_g1_f32_dev_);
  if (tc_g1_f16_dev_ != nullptr) cudaFree(tc_g1_f16_dev_);
  if (tc_c_fp8_dev_ != nullptr) cudaFree(tc_c_fp8_dev_);
  if (tc_c_scale_dev_ != nullptr) cudaFree(tc_c_scale_dev_);
  if (tc_c_bf16_dev_ != nullptr) cudaFree(tc_c_bf16_dev_);
  if (tc_b_bf16_dev_ != nullptr) cudaFree(tc_b_bf16_dev_);
  if (tc_d_f32_dev_ != nullptr) cudaFree(tc_d_f32_dev_);
  if (tc_m_indptr_dev_ != nullptr) cudaFree(tc_m_indptr_dev_);
  if (tc_int_workspace_dev_ != nullptr) cudaFree(tc_int_workspace_dev_);
  if (tc_float_workspace_dev_ != nullptr) cudaFree(tc_float_workspace_dev_);

  tc_max_rows_ = padded_rows;
  constexpr size_t kCutlassWorkspaceBytes = 32ull * 1024ull * 1024ull;
  cudaError_t e = cudaSuccess;
  e = cudaMalloc(&tc_a_fp8_dev_, static_cast<size_t>(padded_rows) * hidden_ * sizeof(uint8_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_a_fp8_dev_");
  e = cudaMalloc(&tc_b_col_dev_, static_cast<size_t>(gemm1_out_) * hidden_ * sizeof(uint8_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_b_col_dev_");
  e = cudaMalloc(&tc_a_scale_dev_, static_cast<size_t>(padded_rows) * hidden_blocks_ * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_a_scale_dev_");
  size_t tc_b_scale_elems =
      std::max(static_cast<size_t>(gemm1_out_blocks_) * hidden_blocks_,
               static_cast<size_t>(hidden_) * intermediate_blocks_);
  e = cudaMalloc(&tc_b_scale_dev_, tc_b_scale_elems * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_b_scale_dev_");
  e = cudaMalloc(&tc_g1_f32_dev_, static_cast<size_t>(padded_rows) * gemm1_out_ * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_g1_f32_dev_");
  e = cudaMalloc(&tc_g1_f16_dev_, static_cast<size_t>(padded_rows) * gemm1_out_ * sizeof(uint16_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_g1_f16_dev_");
  e = cudaMalloc(&tc_c_fp8_dev_, static_cast<size_t>(padded_rows) * intermediate_ * sizeof(uint8_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_c_fp8_dev_");
  e = cudaMalloc(&tc_c_scale_dev_, static_cast<size_t>(padded_rows) * intermediate_blocks_ * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_c_scale_dev_");
  e = cudaMalloc(&tc_c_bf16_dev_, static_cast<size_t>(padded_rows) * intermediate_ * sizeof(uint16_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_c_bf16_dev_");
  e = cudaMalloc(&tc_b_bf16_dev_, static_cast<size_t>(hidden_) * intermediate_ * sizeof(uint16_t));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_b_bf16_dev_");
  e = cudaMalloc(&tc_d_f32_dev_, static_cast<size_t>(padded_rows) * hidden_ * sizeof(float));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_d_f32_dev_");
  e = cudaMalloc(&tc_m_indptr_dev_, 2 * sizeof(int));
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_m_indptr_dev_");
  e = cudaMalloc(&tc_int_workspace_dev_, kCutlassWorkspaceBytes);
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_int_workspace_dev_");
  e = cudaMalloc(&tc_float_workspace_dev_, kCutlassWorkspaceBytes);
  if (e != cudaSuccess) throw std::runtime_error("cudaMalloc failed for tc_float_workspace_dev_");
}

void DeviceMxfpGemmModule::RunExpert(const float* a_dev, int64_t t, const float* local_weight_dev,
                                     int local_expert_idx, const uint8_t* gemm1_w_dev,
                                     const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
                                     const float* gemm2_s_dev, float* out_acc_dev,
                                     cudaStream_t stream) const {
  size_t w13_elems = static_cast<size_t>(gemm1_out_) * static_cast<size_t>(hidden_);
  size_t w13s_elems = static_cast<size_t>(gemm1_out_blocks_) * static_cast<size_t>(hidden_blocks_);
  size_t w2_elems = static_cast<size_t>(hidden_) * static_cast<size_t>(intermediate_);
  size_t w2s_elems = static_cast<size_t>(hidden_blocks_) * static_cast<size_t>(intermediate_blocks_);

  const uint8_t* w13_e = gemm1_w_dev + static_cast<size_t>(local_expert_idx) * w13_elems;
  const float* s13_e = gemm1_s_dev + static_cast<size_t>(local_expert_idx) * w13s_elems;
  const uint8_t* w2_e = gemm2_w_dev + static_cast<size_t>(local_expert_idx) * w2_elems;
  const float* s2_e = gemm2_s_dev + static_cast<size_t>(local_expert_idx) * w2s_elems;
  constexpr int kThreads = 128;
  int64_t n_g1 = t * gemm1_out_;
  int64_t n_c = t * intermediate_;
  int64_t n_out = t * hidden_;

  int b1 = static_cast<int>((n_g1 + kThreads - 1) / kThreads);
  int b2 = static_cast<int>((n_c + kThreads - 1) / kThreads);
  int b3 = static_cast<int>((n_out + kThreads - 1) / kThreads);

  gemm1_kernel<<<b1, kThreads, 0, stream>>>(a_dev, t, hidden_, gemm1_out_, block_, hidden_blocks_,
                                            local_expert_idx, local_weight_dev, w13_e, s13_e,
                                            emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_, g1_dev_);
  swiglu_kernel<<<b2, kThreads, 0, stream>>>(g1_dev_, t, intermediate_, local_expert_idx,
                                             local_weight_dev, emulate_fp8_unit_, c_dev_);
  gemm2_acc_kernel<<<b3, kThreads, 0, stream>>>(c_dev_, t, hidden_, intermediate_, block_,
                                                intermediate_blocks_, local_expert_idx,
                                                local_weight_dev, w2_e, s2_e, emulate_fp8_unit_,
                                                emulate_fp16_operands_,
                                                emulate_acc_half_,
                                                out_acc_dev);
}

void DeviceMxfpGemmModule::RunExpertPermuted(const float* a_dev, int64_t /*t*/, int n_rows,
                                             const int* permuted_tok_e,
                                             const float* permuted_w_e, int local_expert_idx,
                                             const uint8_t* gemm1_w_dev, const float* gemm1_s_dev,
                                             const uint8_t* gemm2_w_dev, const float* gemm2_s_dev,
                                             float* out_acc_dev, cudaStream_t stream) const {
  if (n_rows <= 0) return;

  size_t w13_elems = static_cast<size_t>(gemm1_out_) * static_cast<size_t>(hidden_);
  size_t w13s_elems = static_cast<size_t>(gemm1_out_blocks_) * static_cast<size_t>(hidden_blocks_);
  size_t w2_elems = static_cast<size_t>(hidden_) * static_cast<size_t>(intermediate_);
  size_t w2s_elems = static_cast<size_t>(hidden_blocks_) * static_cast<size_t>(intermediate_blocks_);

  const uint8_t* w13_e = gemm1_w_dev + static_cast<size_t>(local_expert_idx) * w13_elems;
  const float* s13_e = gemm1_s_dev + static_cast<size_t>(local_expert_idx) * w13s_elems;
  const uint8_t* w2_e = gemm2_w_dev + static_cast<size_t>(local_expert_idx) * w2_elems;
  const float* s2_e = gemm2_s_dev + static_cast<size_t>(local_expert_idx) * w2s_elems;

  constexpr int kThreads = 128;
  int64_t n_g1 = static_cast<int64_t>(n_rows) * gemm1_out_;
  int64_t n_c = static_cast<int64_t>(n_rows) * intermediate_;
  int64_t n_out = static_cast<int64_t>(n_rows) * hidden_;

  int b1 = static_cast<int>((n_g1 + kThreads - 1) / kThreads);
  int b2 = static_cast<int>((n_c + kThreads - 1) / kThreads);
  int b3 = static_cast<int>((n_out + kThreads - 1) / kThreads);

  gemm1_permuted_kernel<<<b1, kThreads, 0, stream>>>(
      a_dev, hidden_, gemm1_out_, block_, hidden_blocks_, n_rows, permuted_tok_e, w13_e, s13_e,
      emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_, g1_dev_);
  swiglu_permuted_kernel<<<b2, kThreads, 0, stream>>>(g1_dev_, intermediate_, n_rows,
                                                      emulate_fp8_unit_, c_dev_);
  gemm2_scatter_accumulate_kernel<<<b3, kThreads, 0, stream>>>(
      c_dev_, hidden_, intermediate_, block_, intermediate_blocks_, n_rows, permuted_tok_e,
      permuted_w_e, w2_e, s2_e, emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_,
      out_acc_dev);
}

void DeviceMxfpGemmModule::RunExpertPermutedTc(const uint8_t* hidden_fp8_dev,
                                               const float* hidden_scale_dev, int64_t t,
                                               int n_rows, const int* permuted_tok_e,
                                               const float* permuted_w_e, int local_expert_idx,
                                               const uint8_t* gemm1_w_dev,
                                               const float* gemm1_s_dev,
                                               const uint8_t* gemm2_w_dev,
                                               const float* gemm2_s_dev,
                                               float* out_acc_dev, cudaStream_t stream) {
  if (n_rows <= 0) return;
  if (!SupportsTcPath()) {
    return RunExpertPermuted(nullptr, t, n_rows, permuted_tok_e, permuted_w_e, local_expert_idx,
                             gemm1_w_dev, gemm1_s_dev, gemm2_w_dev, gemm2_s_dev, out_acc_dev,
                             stream);
  }

#if FIB_HAS_DIRECT_CUTLASS_SM100
  EnsureTcWorkspace(n_rows);
  int padded_rows = (n_rows + 3) & ~3;

  size_t w13_elems = static_cast<size_t>(gemm1_out_) * static_cast<size_t>(hidden_);
  size_t w13s_elems = static_cast<size_t>(gemm1_out_blocks_) * static_cast<size_t>(hidden_blocks_);
  size_t w2_elems = static_cast<size_t>(hidden_) * static_cast<size_t>(intermediate_);
  size_t w2s_elems = static_cast<size_t>(hidden_blocks_) * static_cast<size_t>(intermediate_blocks_);

  const uint8_t* w13_e = gemm1_w_dev + static_cast<size_t>(local_expert_idx) * w13_elems;
  const float* s13_e = gemm1_s_dev + static_cast<size_t>(local_expert_idx) * w13s_elems;
  const uint8_t* w2_e = gemm2_w_dev + static_cast<size_t>(local_expert_idx) * w2_elems;
  const float* s2_e = gemm2_s_dev + static_cast<size_t>(local_expert_idx) * w2s_elems;
  // Standalone contract probing shows the direct CUTLASS launcher expects the
  // raw [N, K] payload as-is here. Keep transpose only as an explicit debug
  // override.
  const char* env_force_transpose_b = std::getenv("FIB_MOE_TC_FORCE_TRANSPOSE_B");
  const bool transpose_b =
      (env_force_transpose_b != nullptr && env_force_transpose_b[0] == '1');
  const char* env_scale_major_k = std::getenv("FIB_MOE_TC_SCALE_MAJOR_K");
  const bool scale_major_k = (env_scale_major_k == nullptr || env_scale_major_k[0] == '1');
  const char* env_c_scale_gran_m = std::getenv("FIB_MOE_TC_C_SCALE_GRAN_M");
  const int c_scale_gran_m =
      (env_c_scale_gran_m != nullptr && std::atoi(env_c_scale_gran_m) == 128) ? 128 : 1;

  using FP8 = cutlass::float_e4m3_t;

  constexpr int kThreads = 256;
  int64_t a_elems = static_cast<int64_t>(padded_rows) * hidden_;
  int64_t a_scale_elems = static_cast<int64_t>(padded_rows) * hidden_blocks_;
  gather_hidden_fp8_rows_kernel<<<static_cast<int>((a_elems + kThreads - 1) / kThreads),
                                  kThreads, 0, stream>>>(
      hidden_fp8_dev, permuted_tok_e, n_rows, padded_rows, hidden_, tc_a_fp8_dev_);
  if (scale_major_k) {
    gather_hidden_scale_rows_mn_major_kernel<<<
        static_cast<int>((a_scale_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        hidden_scale_dev, permuted_tok_e, t, n_rows, padded_rows, hidden_blocks_,
        tc_a_scale_dev_);
  } else {
    gather_hidden_scale_rows_kernel<<<static_cast<int>((a_scale_elems + kThreads - 1) / kThreads),
                                      kThreads, 0, stream>>>(
        hidden_scale_dev, permuted_tok_e, t, n_rows, padded_rows, hidden_blocks_,
        tc_a_scale_dev_);
  }
  write_single_group_indptr_kernel<<<1, 1, 0, stream>>>(padded_rows, tc_m_indptr_dev_);
  int64_t w13_transpose_elems = static_cast<int64_t>(gemm1_out_) * hidden_;
  if (transpose_b) {
    transpose_rowmajor_nk_to_colmajor_nk_kernel<<<
        static_cast<int>((w13_transpose_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        w13_e, gemm1_out_, hidden_, tc_b_col_dev_);
  }
  FP8* w13_tc = transpose_b ? reinterpret_cast<FP8*>(tc_b_col_dev_)
                            : const_cast<FP8*>(reinterpret_cast<const FP8*>(w13_e));
  int w13_scale_elems = gemm1_out_blocks_ * hidden_blocks_;
  if (scale_major_k) {
    copy_scale_nblock_kblock_kernel<<<(w13_scale_elems + kThreads - 1) / kThreads, kThreads, 0,
                                      stream>>>(s13_e, w13_scale_elems, tc_b_scale_dev_);
  } else {
    transpose_scale_nblock_kblock_to_kblock_nblock_kernel<<<
        (w13_scale_elems + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        s13_e, gemm1_out_blocks_, hidden_blocks_, tc_b_scale_dev_);
  }

  auto run_gemm1_1sm = [&]() {
    if (scale_major_k) {
      return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 1, cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_), w13_tc,
          tc_a_scale_dev_, tc_b_scale_dev_, tc_g1_f32_dev_,
          padded_rows, gemm1_out_, hidden_, stream);
    }
    return launch_cutlass_blockscaled_group_gemm_sm100<1, false, 1, cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_), w13_tc,
        tc_a_scale_dev_, tc_b_scale_dev_, tc_g1_f32_dev_,
        padded_rows, gemm1_out_, hidden_, stream);
  };
  auto run_gemm1_2sm = [&]() {
    if (scale_major_k) {
      return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 2, cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_), w13_tc,
          tc_a_scale_dev_, tc_b_scale_dev_, tc_g1_f32_dev_,
          padded_rows, gemm1_out_, hidden_, stream);
    }
    return launch_cutlass_blockscaled_group_gemm_sm100<1, false, 2, cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_a_fp8_dev_), w13_tc,
        tc_a_scale_dev_, tc_b_scale_dev_, tc_g1_f32_dev_,
        padded_rows, gemm1_out_, hidden_, stream);
  };
  cudaError_t st1 = (padded_rows >= 256) ? run_gemm1_2sm() : run_gemm1_1sm();
  if (st1 != cudaSuccess) return;

  const char* env_compare_g1 = std::getenv("FIB_MOE_TC_COMPARE_G1");
  if (env_compare_g1 != nullptr && env_compare_g1[0] == '1') {
    EnsureWorkspace(n_rows, stream);
    int64_t g1_elems = static_cast<int64_t>(n_rows) * gemm1_out_;
    int64_t g1_bytes = g1_elems * static_cast<int64_t>(sizeof(float));
    int64_t a_elems_f32 = static_cast<int64_t>(n_rows) * hidden_;
    int64_t a_bytes_f32 = a_elems_f32 * static_cast<int64_t>(sizeof(float));
    if (a_bytes_f32 + g1_bytes + static_cast<int64_t>(sizeof(float)) <= 32ll * 1024ll * 1024ll) {
      float* a_ref_dev = reinterpret_cast<float*>(tc_int_workspace_dev_);
      float* g1_ref_dev = reinterpret_cast<float*>(tc_float_workspace_dev_);
      float* g1_max_diff_dev =
          reinterpret_cast<float*>(reinterpret_cast<char*>(tc_float_workspace_dev_) + g1_bytes);
      dequant_fp8_rows_kernel<<<static_cast<int>((a_elems_f32 + kThreads - 1) / kThreads), kThreads,
                                0, stream>>>(tc_a_fp8_dev_, tc_a_scale_dev_, n_rows, hidden_, 1,
                                             hidden_blocks_, scale_major_k, a_ref_dev);
      gemm1_compact_kernel<<<static_cast<int>((g1_elems + kThreads - 1) / kThreads), kThreads, 0,
                             stream>>>(
          a_ref_dev, hidden_, gemm1_out_, block_, hidden_blocks_, n_rows, w13_e, s13_e,
          emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_, g1_ref_dev);
      cudaMemcpyAsync(g1_dev_, tc_g1_f32_dev_, g1_elems * sizeof(float), cudaMemcpyDeviceToDevice,
                      stream);
      cudaMemsetAsync(g1_max_diff_dev, 0, sizeof(float), stream);
      compare_abs_diff_kernel<<<static_cast<int>((g1_elems + kThreads - 1) / kThreads), kThreads,
                                0, stream>>>(tc_g1_f32_dev_, g1_ref_dev, g1_elems,
                                             g1_max_diff_dev);
      float g1_max_diff = 0.0f;
      cudaMemcpyAsync(&g1_max_diff, g1_max_diff_dev, sizeof(float), cudaMemcpyDeviceToHost,
                      stream);
      cudaStreamSynchronize(stream);
      std::fprintf(stderr, "[mxfp] local_expert=%d n_rows=%d g1_max_abs_diff=%g\n",
                   local_expert_idx, n_rows, g1_max_diff);
      std::fflush(stderr);
    }
  }

  if (std::getenv("FIB_MOE_TC_GEMM1_ONLY") != nullptr) {
    EnsureWorkspace(n_rows, stream);
    int64_t g1_elems = static_cast<int64_t>(n_rows) * gemm1_out_;
    float_to_f16_kernel<<<static_cast<int>((g1_elems + kThreads - 1) / kThreads), kThreads, 0,
                          stream>>>(tc_g1_f32_dev_, g1_elems, tc_g1_f16_dev_);
    f16_matrix_to_float_kernel<<<static_cast<int>((g1_elems + kThreads - 1) / kThreads), kThreads,
                                 0, stream>>>(tc_g1_f16_dev_, g1_elems, g1_dev_);
    int64_t c_elems = static_cast<int64_t>(n_rows) * intermediate_;
    int64_t out_elems = static_cast<int64_t>(n_rows) * hidden_;
    swiglu_permuted_kernel<<<static_cast<int>((c_elems + kThreads - 1) / kThreads),
                              kThreads, 0, stream>>>(
        g1_dev_, intermediate_, n_rows, false, c_dev_);
    gemm2_scatter_accumulate_kernel<<<static_cast<int>((out_elems + kThreads - 1) / kThreads),
                                      kThreads, 0, stream>>>(
        c_dev_, hidden_, intermediate_, block_, intermediate_blocks_, n_rows, permuted_tok_e,
        permuted_w_e, w2_e, s2_e, emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_,
        out_acc_dev);
    return;
  }

  EnsureWorkspace(n_rows, stream);
  int64_t g1_elems = static_cast<int64_t>(n_rows) * gemm1_out_;
  float_to_f16_kernel<<<static_cast<int>((g1_elems + kThreads - 1) / kThreads), kThreads, 0,
                        stream>>>(tc_g1_f32_dev_, g1_elems, tc_g1_f16_dev_);
  f16_matrix_to_float_kernel<<<static_cast<int>((g1_elems + kThreads - 1) / kThreads), kThreads, 0,
                               stream>>>(tc_g1_f16_dev_, g1_elems, g1_dev_);
  int64_t c_elems = static_cast<int64_t>(n_rows) * intermediate_;
  swiglu_permuted_kernel<<<static_cast<int>((c_elems + kThreads - 1) / kThreads), kThreads, 0,
                           stream>>>(g1_dev_, intermediate_, n_rows, false, c_dev_);
  const char* env_gemm2_f16 = std::getenv("FIB_MOE_TC_GEMM2_F16");
  const char* env_gemm2_bf16 = std::getenv("FIB_MOE_TC_GEMM2_BF16");
  const bool gemm2_f16 = (env_gemm2_f16 != nullptr && env_gemm2_f16[0] == '1');
  const bool gemm2_bf16 =
      !gemm2_f16 && (env_gemm2_bf16 == nullptr || env_gemm2_bf16[0] != '0');
  if (gemm2_bf16) {
    float_to_bf16_kernel<<<static_cast<int>((c_elems + kThreads - 1) / kThreads), kThreads, 0,
                           stream>>>(c_dev_, c_elems, tc_c_bf16_dev_);
  } else if (gemm2_f16) {
    float_rows_scaled_to_f16_kernel<<<padded_rows, 256, 0, stream>>>(
        c_dev_, intermediate_, n_rows, padded_rows, tc_c_bf16_dev_, tc_c_scale_dev_);
  } else {
    if (c_scale_gran_m == 128) {
      int c_row_blocks = (padded_rows + 127) / 128;
      dim3 qgrid(c_row_blocks, intermediate_blocks_);
      compute_block_scale_128x128_kernel<<<qgrid, 256, 256 * sizeof(float), stream>>>(
          c_dev_, intermediate_, n_rows, padded_rows, c_row_blocks, intermediate_blocks_,
          tc_c_scale_dev_, scale_major_k);
      quantize_float_blocks_128x128_to_fp8_kernel<<<
          static_cast<int>(((static_cast<int64_t>(padded_rows) * intermediate_) + kThreads - 1) /
                           kThreads),
          kThreads, 0, stream>>>(c_dev_, tc_c_scale_dev_, intermediate_, n_rows, padded_rows,
                                 c_row_blocks, intermediate_blocks_, tc_c_fp8_dev_, scale_major_k);
    } else {
      dim3 qgrid(padded_rows, intermediate_blocks_);
      quantize_float_rows_to_fp8_mse_kernel<<<qgrid, 128, 0, stream>>>(
          c_dev_, intermediate_, n_rows, padded_rows, tc_c_fp8_dev_, tc_c_scale_dev_,
          scale_major_k);
    }
  }
  const char* env_compare_c = std::getenv("FIB_MOE_TC_COMPARE_C");
  if (!gemm2_bf16 && !gemm2_f16 && env_compare_c != nullptr && env_compare_c[0] == '1') {
    int64_t c_all_elems = static_cast<int64_t>(n_rows) * intermediate_;
    int64_t c_all_bytes = c_all_elems * static_cast<int64_t>(sizeof(float));
    int64_t g1_ref_elems = static_cast<int64_t>(n_rows) * gemm1_out_;
    int64_t g1_ref_bytes = g1_ref_elems * static_cast<int64_t>(sizeof(float));
    int64_t a_ref_elems = static_cast<int64_t>(n_rows) * hidden_;
    int64_t a_ref_bytes = a_ref_elems * static_cast<int64_t>(sizeof(float));
    int64_t need_bytes = 2 * c_all_bytes + 2 * static_cast<int64_t>(sizeof(float));
    if (need_bytes <= 32ll * 1024ll * 1024ll &&
        a_ref_bytes + g1_ref_bytes <= 32ll * 1024ll * 1024ll) {
      float* a_ref_dev = reinterpret_cast<float*>(tc_int_workspace_dev_);
      float* g1_ref_dev =
          reinterpret_cast<float*>(reinterpret_cast<char*>(tc_int_workspace_dev_) + a_ref_bytes);
      float* c_ref_dev = reinterpret_cast<float*>(tc_float_workspace_dev_);
      float* c_deq_dev = reinterpret_cast<float*>(reinterpret_cast<char*>(tc_float_workspace_dev_) +
                                                  c_all_bytes);
      float* c_cur_max_diff_dev =
          reinterpret_cast<float*>(reinterpret_cast<char*>(tc_float_workspace_dev_) +
                                   2 * c_all_bytes);
      float* c_q_max_diff_dev =
          reinterpret_cast<float*>(reinterpret_cast<char*>(tc_float_workspace_dev_) +
                                   2 * c_all_bytes + sizeof(float));
      dequant_fp8_rows_kernel<<<static_cast<int>((a_ref_elems + kThreads - 1) / kThreads), kThreads,
                                0, stream>>>(tc_a_fp8_dev_, tc_a_scale_dev_, n_rows, hidden_, 1,
                                             hidden_blocks_, scale_major_k, a_ref_dev);
      gemm1_compact_kernel<<<static_cast<int>((g1_ref_elems + kThreads - 1) / kThreads), kThreads,
                             0, stream>>>(
          a_ref_dev, hidden_, gemm1_out_, block_, hidden_blocks_, n_rows, w13_e, s13_e,
          emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_, g1_ref_dev);
      swiglu_permuted_kernel<<<static_cast<int>((c_all_elems + kThreads - 1) / kThreads), kThreads,
                               0, stream>>>(g1_ref_dev, intermediate_, n_rows, false, c_ref_dev);
      dequant_fp8_rows_native_kernel<<<
          static_cast<int>((c_all_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
          tc_c_fp8_dev_, tc_c_scale_dev_, n_rows, intermediate_, c_scale_gran_m,
          intermediate_blocks_,
          scale_major_k, c_deq_dev);
      cudaMemsetAsync(c_cur_max_diff_dev, 0, sizeof(float), stream);
      cudaMemsetAsync(c_q_max_diff_dev, 0, sizeof(float), stream);
      compare_abs_diff_kernel<<<static_cast<int>((c_all_elems + kThreads - 1) / kThreads), kThreads,
                                0, stream>>>(c_ref_dev, c_dev_, c_all_elems, c_cur_max_diff_dev);
      compare_abs_diff_kernel<<<static_cast<int>((c_all_elems + kThreads - 1) / kThreads), kThreads,
                                0, stream>>>(c_ref_dev, c_deq_dev, c_all_elems, c_q_max_diff_dev);
      float c_cur_max_diff = 0.0f;
      float c_q_max_diff = 0.0f;
      cudaMemcpyAsync(&c_cur_max_diff, c_cur_max_diff_dev, sizeof(float), cudaMemcpyDeviceToHost,
                      stream);
      cudaMemcpyAsync(&c_q_max_diff, c_q_max_diff_dev, sizeof(float), cudaMemcpyDeviceToHost,
                      stream);
      cudaStreamSynchronize(stream);
      std::fprintf(stderr,
                   "[mxfp] local_expert=%d n_rows=%d c_float_max_abs_diff=%g c_quant_max_abs_diff=%g\n",
                   local_expert_idx, n_rows, c_cur_max_diff, c_q_max_diff);
      const char* env_dump_c_block_errors = std::getenv("FIB_MOE_TC_DUMP_C_BLOCK_ERRORS");
      if (env_dump_c_block_errors != nullptr && env_dump_c_block_errors[0] == '1' && n_rows > 0 &&
          c_scale_gran_m == 1) {
        std::vector<float> c_ref_host(static_cast<size_t>(c_all_elems));
        std::vector<float> c_deq_host(static_cast<size_t>(c_all_elems));
        std::vector<float> scale_host(static_cast<size_t>(n_rows) * intermediate_blocks_);
        cudaMemcpyAsync(c_ref_host.data(), c_ref_dev, c_all_bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(c_deq_host.data(), c_deq_dev, c_all_bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(scale_host.data(), tc_c_scale_dev_,
                        scale_host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        struct BlockErr {
          int row;
          int cb;
          float max_abs;
          float mean_abs;
          float scale;
        };
        std::array<BlockErr, 4> top_blocks{};
        for (auto& e : top_blocks) {
          e.row = -1;
          e.cb = -1;
          e.max_abs = -1.0f;
          e.mean_abs = -1.0f;
          e.scale = 0.0f;
        }

        for (int row = 0; row < n_rows; ++row) {
          for (int cb = 0; cb < intermediate_blocks_; ++cb) {
            float max_abs = 0.0f;
            float sum_abs = 0.0f;
            int col0 = cb * 128;
            for (int u = 0; u < 128; ++u) {
              int col = col0 + u;
              float diff = std::fabs(
                  c_ref_host[static_cast<size_t>(row) * intermediate_ + col] -
                  c_deq_host[static_cast<size_t>(row) * intermediate_ + col]);
              max_abs = std::max(max_abs, diff);
              sum_abs += diff;
            }
            BlockErr cur{row, cb, max_abs, sum_abs / 128.0f,
                         scale_major_k ? scale_host[static_cast<size_t>(row) * intermediate_blocks_ + cb]
                                       : scale_host[static_cast<size_t>(cb) * n_rows + row]};
            for (int slot = 0; slot < static_cast<int>(top_blocks.size()); ++slot) {
              if (cur.max_abs > top_blocks[slot].max_abs) {
                for (int j = static_cast<int>(top_blocks.size()) - 1; j > slot; --j) {
                  top_blocks[j] = top_blocks[j - 1];
                }
                top_blocks[slot] = cur;
                break;
              }
            }
          }
        }
        for (const auto& e : top_blocks) {
          if (e.row < 0) break;
          std::fprintf(stderr,
                       "[mxfp] c_block_err expert=%d row=%d cb=%d scale=%g max_abs=%g mean_abs=%g\n",
                       local_expert_idx, e.row, e.cb, e.scale, e.max_abs, e.mean_abs);
        }
      }
      const char* env_dump_c_block = std::getenv("FIB_MOE_TC_DUMP_C_BLOCK");
      if (env_dump_c_block != nullptr && env_dump_c_block[0] == '1' && n_rows > 0) {
        float c_ref_host[8] = {0};
        float c_cur_host[8] = {0};
        float c_deq_host[8] = {0};
        uint8_t q_host[8] = {0};
        float scale_host[4] = {0};
        cudaMemcpyAsync(c_ref_host, c_ref_dev, sizeof(c_ref_host), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(c_cur_host, c_dev_, sizeof(c_cur_host), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(c_deq_host, c_deq_dev, sizeof(c_deq_host), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(q_host, tc_c_fp8_dev_, sizeof(q_host), cudaMemcpyDeviceToHost, stream);
        int scale_cols = intermediate_blocks_;
        if (c_scale_gran_m == 128) {
          int row_blocks = (padded_rows + 127) / 128;
          int to_copy = std::min(scale_cols, 4);
          if (scale_major_k) {
            cudaMemcpyAsync(scale_host, tc_c_scale_dev_, to_copy * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
          } else {
            cudaMemcpyAsync(scale_host, tc_c_scale_dev_, to_copy * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
          }
          (void)row_blocks;
        } else {
          int to_copy = std::min(scale_cols, 4);
          if (scale_major_k) {
            cudaMemcpyAsync(scale_host, tc_c_scale_dev_, to_copy * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
          } else {
            cudaMemcpyAsync(scale_host, tc_c_scale_dev_, to_copy * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
          }
        }
        cudaStreamSynchronize(stream);
        std::fprintf(stderr,
                     "[mxfp] dump_c_block expert=%d scale_gran_m=%d scale_major_k=%d "
                     "scale0-3=[%g %g %g %g]\n",
                     local_expert_idx, c_scale_gran_m, scale_major_k ? 1 : 0, scale_host[0],
                     scale_host[1], scale_host[2], scale_host[3]);
        std::fprintf(stderr,
                     "[mxfp] dump_c_block ref0-7=[%g %g %g %g %g %g %g %g]\n",
                     c_ref_host[0], c_ref_host[1], c_ref_host[2], c_ref_host[3], c_ref_host[4],
                     c_ref_host[5], c_ref_host[6], c_ref_host[7]);
        std::fprintf(stderr,
                     "[mxfp] dump_c_block cur0-7=[%g %g %g %g %g %g %g %g]\n",
                     c_cur_host[0], c_cur_host[1], c_cur_host[2], c_cur_host[3], c_cur_host[4],
                     c_cur_host[5], c_cur_host[6], c_cur_host[7]);
        std::fprintf(stderr,
                     "[mxfp] dump_c_block q0-7=[%u %u %u %u %u %u %u %u]\n", q_host[0], q_host[1],
                     q_host[2], q_host[3], q_host[4], q_host[5], q_host[6], q_host[7]);
        std::fprintf(stderr,
                     "[mxfp] dump_c_block deq0-7=[%g %g %g %g %g %g %g %g]\n",
                     c_deq_host[0], c_deq_host[1], c_deq_host[2], c_deq_host[3], c_deq_host[4],
                     c_deq_host[5], c_deq_host[6], c_deq_host[7]);
      }
      std::fflush(stderr);
    }
  }
  if (std::getenv("FIB_MOE_TC_GEMM2_REF") != nullptr) {
    if (!gemm2_bf16 && !gemm2_f16) {
      dequant_fp8_rows_native_kernel<<<
          static_cast<int>((c_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
          tc_c_fp8_dev_, tc_c_scale_dev_, n_rows, intermediate_, c_scale_gran_m,
          intermediate_blocks_, scale_major_k, c_dev_);
    }
    int64_t out_elems = static_cast<int64_t>(n_rows) * hidden_;
    gemm2_scatter_accumulate_kernel<<<static_cast<int>((out_elems + kThreads - 1) / kThreads),
                                      kThreads, 0, stream>>>(
        c_dev_, hidden_, intermediate_, block_, intermediate_blocks_, n_rows, permuted_tok_e,
        permuted_w_e, w2_e, s2_e, emulate_fp8_unit_, emulate_fp16_operands_, emulate_acc_half_,
        out_acc_dev);
    return;
  }
  int64_t w2_transpose_elems = static_cast<int64_t>(hidden_) * intermediate_;
  if (transpose_b) {
    transpose_rowmajor_nk_to_colmajor_nk_kernel<<<
        static_cast<int>((w2_transpose_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        w2_e, hidden_, intermediate_, tc_b_col_dev_);
  }
  FP8* w2_tc = transpose_b ? reinterpret_cast<FP8*>(tc_b_col_dev_)
                           : const_cast<FP8*>(reinterpret_cast<const FP8*>(w2_e));
  int w2_scale_elems = hidden_blocks_ * intermediate_blocks_;
  if (gemm2_bf16) {
    dequant_fp8_rows_native_to_bf16_kernel<<<
        static_cast<int>((w2_transpose_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        w2_e, s2_e, hidden_, intermediate_, 128, intermediate_blocks_, true, tc_b_bf16_dev_);
  } else if (gemm2_f16) {
    dequant_fp8_rows_native_to_f16_kernel<<<
        static_cast<int>((w2_transpose_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        w2_e, s2_e, hidden_, intermediate_, 128, intermediate_blocks_, true, tc_b_bf16_dev_);
  } else if (scale_major_k) {
    copy_scale_nblock_kblock_kernel<<<(w2_scale_elems + kThreads - 1) / kThreads, kThreads, 0,
                                      stream>>>(s2_e, w2_scale_elems, tc_b_scale_dev_);
  } else {
    transpose_scale_nblock_kblock_to_kblock_nblock_kernel<<<
        (w2_scale_elems + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        s2_e, hidden_blocks_, intermediate_blocks_, tc_b_scale_dev_);
  }

  auto run_gemm2_1sm = [&]() {
    if (gemm2_bf16) {
      return launch_cutlass_dense_gemm_sm100<1, cutlass::bfloat16_t, cutlass::bfloat16_t>(
          tc_float_workspace_dev_, 32ull * 1024ull * 1024ull,
          reinterpret_cast<cutlass::bfloat16_t*>(tc_c_bf16_dev_),
          reinterpret_cast<cutlass::bfloat16_t*>(tc_b_bf16_dev_), tc_d_f32_dev_, padded_rows, hidden_,
          intermediate_, stream);
    }
    if (gemm2_f16) {
      return launch_cutlass_dense_gemm_sm100<1, cutlass::half_t, cutlass::half_t>(
          tc_float_workspace_dev_, 32ull * 1024ull * 1024ull,
          reinterpret_cast<cutlass::half_t*>(tc_c_bf16_dev_),
          reinterpret_cast<cutlass::half_t*>(tc_b_bf16_dev_), tc_d_f32_dev_, padded_rows, hidden_,
          intermediate_, stream);
    }
    if (scale_major_k) {
      if (c_scale_gran_m == 128) {
        return launch_cutlass_blockscaled_group_gemm_sm100<128, true, 1,
                                                           cutlass::float_e4m3_t,
                                                           cutlass::float_e4m3_t, float>(
            tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
            32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
            tc_c_scale_dev_, tc_b_scale_dev_, tc_d_f32_dev_, padded_rows, hidden_,
            intermediate_, stream);
      }
      return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 1, cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
          tc_c_scale_dev_, tc_b_scale_dev_, tc_d_f32_dev_,
          padded_rows, hidden_, intermediate_, stream);
    }
    if (c_scale_gran_m == 128) {
      return launch_cutlass_blockscaled_group_gemm_sm100<128, false, 1,
                                                         cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
          tc_c_scale_dev_, tc_b_scale_dev_, tc_d_f32_dev_,
          padded_rows, hidden_, intermediate_, stream);
    }
    return launch_cutlass_blockscaled_group_gemm_sm100<1, false, 1, cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
        tc_c_scale_dev_, tc_b_scale_dev_, tc_d_f32_dev_,
        padded_rows, hidden_, intermediate_, stream);
  };
  auto run_gemm2_2sm = [&]() {
    if (gemm2_bf16) {
      return launch_cutlass_dense_gemm_sm100<2, cutlass::bfloat16_t, cutlass::bfloat16_t>(
          tc_float_workspace_dev_, 32ull * 1024ull * 1024ull,
          reinterpret_cast<cutlass::bfloat16_t*>(tc_c_bf16_dev_),
          reinterpret_cast<cutlass::bfloat16_t*>(tc_b_bf16_dev_), tc_d_f32_dev_, padded_rows, hidden_,
          intermediate_, stream);
    }
    if (gemm2_f16) {
      return launch_cutlass_dense_gemm_sm100<2, cutlass::half_t, cutlass::half_t>(
          tc_float_workspace_dev_, 32ull * 1024ull * 1024ull,
          reinterpret_cast<cutlass::half_t*>(tc_c_bf16_dev_),
          reinterpret_cast<cutlass::half_t*>(tc_b_bf16_dev_), tc_d_f32_dev_, padded_rows, hidden_,
          intermediate_, stream);
    }
    if (scale_major_k) {
      if (c_scale_gran_m == 128) {
        return launch_cutlass_blockscaled_group_gemm_sm100<128, true, 2,
                                                           cutlass::float_e4m3_t,
                                                           cutlass::float_e4m3_t, float>(
            tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
            32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
            tc_c_scale_dev_, tc_b_scale_dev_, tc_d_f32_dev_, padded_rows, hidden_,
            intermediate_, stream);
      }
      return launch_cutlass_blockscaled_group_gemm_sm100<1, true, 2, cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
          tc_c_scale_dev_, tc_b_scale_dev_, tc_d_f32_dev_,
          padded_rows, hidden_, intermediate_, stream);
    }
    if (c_scale_gran_m == 128) {
      return launch_cutlass_blockscaled_group_gemm_sm100<128, false, 2,
                                                         cutlass::float_e4m3_t,
                                                         cutlass::float_e4m3_t, float>(
          tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
          32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
          tc_c_scale_dev_, tc_b_scale_dev_, tc_d_f32_dev_,
          padded_rows, hidden_, intermediate_, stream);
    }
    return launch_cutlass_blockscaled_group_gemm_sm100<1, false, 2, cutlass::float_e4m3_t,
                                                       cutlass::float_e4m3_t, float>(
        tc_int_workspace_dev_, 32ull * 1024ull * 1024ull, tc_float_workspace_dev_,
        32ull * 1024ull * 1024ull, reinterpret_cast<FP8*>(tc_c_fp8_dev_), w2_tc,
        tc_c_scale_dev_, tc_b_scale_dev_, tc_d_f32_dev_,
        padded_rows, hidden_, intermediate_, stream);
  };
  cudaError_t st2 = (padded_rows >= 256) ? run_gemm2_2sm() : run_gemm2_1sm();
  if (st2 != cudaSuccess) return;

  int64_t scatter_elems = static_cast<int64_t>(n_rows) * hidden_;
  if (gemm2_f16) {
    scatter_float_weighted_row_scaled_kernel<<<
        static_cast<int>((scatter_elems + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        tc_d_f32_dev_, tc_c_scale_dev_, hidden_, n_rows, permuted_tok_e, permuted_w_e,
        out_acc_dev);
  } else {
    scatter_float_weighted_kernel<<<static_cast<int>((scatter_elems + kThreads - 1) / kThreads),
                                    kThreads, 0, stream>>>(
        tc_d_f32_dev_, hidden_, n_rows, permuted_tok_e, permuted_w_e, out_acc_dev);
  }
#else
  return RunExpertPermuted(nullptr, t, n_rows, permuted_tok_e, permuted_w_e, local_expert_idx,
                           gemm1_w_dev, gemm1_s_dev, gemm2_w_dev, gemm2_s_dev, out_acc_dev,
                           stream);
#endif
}

}  // namespace mxfp
