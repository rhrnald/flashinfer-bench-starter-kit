#pragma once

#include <cuda_runtime.h>
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if __has_include(<cutlass/cutlass.h>) && __has_include(<cute/tensor.hpp>) && \
    __has_include(<cutlass/gemm/device/gemm_universal_adapter.h>)
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_mixed_dtype_blockwise_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/kernel_hardware_info.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/version.h>
#define FIB_HAS_DIRECT_CUTLASS_SM100 1
#else
#define FIB_HAS_DIRECT_CUTLASS_SM100 0
#endif

// Keep production kernels CUTLASS-only. The FlashInfer grouped SM100 path was
// only an experimental comparison target and is not part of the accepted
// fp16/fp32-scale + e4m3 contract.
#define FIB_HAS_FLASHINFER_GROUP_GEMM_FP8_SM100 0

namespace mxfp::detail {

#if FIB_HAS_DIRECT_CUTLASS_SM100

inline char* align_buffer_ptr(char* ptr, size_t alignment) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
  return reinterpret_cast<char*>(aligned);
}

template <typename T>
T* allocate_aligned_buffer(char*& cursor, size_t& remaining, size_t count, size_t alignment = 16) {
  char* aligned = align_buffer_ptr(cursor, alignment);
  size_t padding = static_cast<size_t>(aligned - cursor);
  size_t bytes = sizeof(T) * count;
  if (padding + bytes > remaining) {
    return nullptr;
  }
  cursor = aligned + bytes;
  remaining -= padding + bytes;
  return reinterpret_cast<T*>(aligned);
}

inline void* allocate_aligned_bytes(char*& cursor, size_t& remaining, size_t bytes,
                                   size_t alignment = 16) {
  char* aligned = align_buffer_ptr(cursor, alignment);
  size_t padding = static_cast<size_t>(aligned - cursor);
  if (padding + bytes > remaining) {
    return nullptr;
  }
  cursor = aligned + bytes;
  remaining -= padding + bytes;
  return aligned;
}

inline cudaError_t cutlass_status_to_cuda(cutlass::Status status) {
  return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

template <int ScaleGranularityM, bool ScaleMajorK, int MmaTileM, int MmaSM, typename DTypeA,
          typename DTypeB, typename DTypeOut>
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

  using ElementC = DTypeOut;
  using LayoutC = LayoutD;
  constexpr int AlignmentC = AlignmentD;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using MmaTileShape = Shape<Int<MmaTileM>, _128, _128>;
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

template <int ScaleGranularityM, int ScaleGranularityN, bool ScaleMajorK, int MmaTileN,
          typename DTypeA, typename DTypeB, typename DTypeOut>
cudaError_t launch_cutlass_blockscaled_group_gemm_tn_sm100(
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
  using LayoutD = cutlass::layout::ColumnMajor;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void;
  using LayoutC = void;
  constexpr int AlignmentC = 0;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using MmaTileShape = Shape<_128, Int<MmaTileN>, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using ScaleConfig = std::conditional_t<
      ScaleMajorK,
      cutlass::detail::Sm1xxBlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, 128,
                                                 UMMA::Major::K, UMMA::Major::K>,
      cutlass::detail::Sm1xxBlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, 128,
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
      {{}, nullptr, {}, d_ptr, stride_d}};
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

template <typename ScaleConfig, typename DTypeIn, typename DTypeSF, typename DTypeC,
          typename DTypeOut, typename ProblemShape, typename StrideA, typename StrideB,
          typename StrideC, typename StrideD, typename LayoutSFA, typename LayoutSFB,
          bool ScaleMajorK>
__global__ void compute_sm1xx_cutlass_group_gemm_args(
    DTypeIn* a, DTypeIn* b, DTypeSF* sfa, DTypeSF* sfb, DTypeOut* d, int* m_indptr,
    int max_m, int n, int k, int num_groups, int scale_granularity_m,
    int scale_granularity_n, int scale_granularity_k, ProblemShape* problem_sizes,
    const DTypeIn** a_ptr, const DTypeIn** b_ptr, const DTypeSF** sfa_ptr,
    const DTypeSF** sfb_ptr, const DTypeC** c_ptr, DTypeOut** d_ptr, StrideA* stride_a,
    StrideB* stride_b, StrideC* stride_c, StrideD* stride_d, LayoutSFA* layout_sfa,
    LayoutSFB* layout_sfb) {
  int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= num_groups) return;
  int m_offset = m_indptr[group];
  int m_next = m_indptr[group + 1];
  int m = m_next - m_offset;
  int sf_n = n / scale_granularity_n;
  int sf_k = k / scale_granularity_k;
  int sf_m_offset = m_offset / scale_granularity_m;

  problem_sizes[group] = ProblemShape(m, n, k);
  stride_a[group] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  stride_b[group] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  // CUTLASS ptr-array grouped examples build InternalStride{C,D} from the
  // logical GEMM output shape (M,N,L). Swapping M/N corrupts non-square groups.
  stride_c[group] = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  stride_d[group] = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
  a_ptr[group] = a + static_cast<int64_t>(m_offset) * k;
  b_ptr[group] = b + static_cast<int64_t>(group) * n * k;
  c_ptr[group] = reinterpret_cast<const DTypeC*>(d + static_cast<int64_t>(m_offset) * n);
  d_ptr[group] = d + static_cast<int64_t>(m_offset) * n;

  if constexpr (ScaleMajorK) {
    layout_sfa[group] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    sfa_ptr[group] = sfa + static_cast<int64_t>(sf_m_offset) * sf_k;
  } else {
    layout_sfa[group] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(max_m, n, k, 1));
    sfa_ptr[group] = sfa + sf_m_offset;
  }
  layout_sfb[group] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  sfb_ptr[group] = sfb + static_cast<int64_t>(group) * sf_n * sf_k;
}

template <int ScaleGranularityM, bool ScaleMajorK, int MmaTileM, int MmaSM, typename DTypeA,
          typename DTypeB, typename DTypeOut, bool UseTmaEpilogue = false,
          int MmaTileN = 128>
cudaError_t launch_cutlass_blockscaled_grouped_ptr_gemm_sm100(
    void* arg_buffer, size_t arg_buffer_size_in_bytes, void* workspace,
    size_t workspace_size_in_bytes, DTypeA* a_ptr_base, DTypeB* b_ptr_base, float* sfa_ptr_base,
    float* sfb_ptr_base, DTypeOut* d_ptr_base, int* m_indptr, int max_m, int n, int k,
    int num_groups, cudaStream_t stream, const int* m_indptr_host = nullptr,
    const int* group_ids_host = nullptr) {
  using namespace cute;
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
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

  using MmaTileShape = Shape<Int<MmaTileM>, Int<MmaTileN>, _128>;
  using ClusterShape = Shape<Int<MmaSM>, _1, _1>;
  using ScaleConfig = std::conditional_t<
      ScaleMajorK,
      cutlass::detail::Sm1xxBlockwiseScaleConfig<ScaleGranularityM, 128, 128, UMMA::Major::K,
                                                 UMMA::Major::K>,
      cutlass::detail::Sm1xxBlockwiseScaleConfig<ScaleGranularityM, 128, 128,
                                                 UMMA::Major::MN, UMMA::Major::MN>>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using NoSmemEpilogueSchedule =
      std::conditional_t<MmaSM == 1,
                         cutlass::epilogue::PtrArrayBlockwiseNoSmemWarpSpecialized1Sm,
                         cutlass::epilogue::PtrArrayBlockwiseNoSmemWarpSpecialized2Sm>;
  using TmaEpilogueSchedule =
      std::conditional_t<MmaSM == 1, cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm,
                         cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>;
  using EpilogueSchedule =
      std::conditional_t<UseTmaEpilogue, TmaEpilogueSchedule, NoSmemEpilogueSchedule>;
  using BaseCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC*, AlignmentC, ElementD, LayoutD*, AlignmentD,
      EpilogueSchedule>::CollectiveOp;
  using CollectiveEpilogue = BaseCollectiveEpilogue;

  using MainloopSchedule =
      std::conditional_t<MmaSM == 1,
                         cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100,
                         cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100>;
  using BaseCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA, ElementB,
      cute::tuple<LayoutB*, LayoutSFB*>, AlignmentB, ElementAccumulator, MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;
  using CollectiveMainloop = BaseCollectiveMainloop;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  char* int_cursor = reinterpret_cast<char*>(arg_buffer);
  size_t int_remaining = arg_buffer_size_in_bytes;
  auto problem_sizes = allocate_aligned_buffer<typename ProblemShape::UnderlyingProblemShape>(
      int_cursor, int_remaining, num_groups);
  auto a_ptr = allocate_aligned_buffer<const typename Gemm::ElementA*>(
      int_cursor, int_remaining, num_groups);
  auto b_ptr = allocate_aligned_buffer<const typename Gemm::ElementB*>(
      int_cursor, int_remaining, num_groups);
  auto c_ptr = allocate_aligned_buffer<const typename Gemm::ElementC*>(int_cursor, int_remaining,
                                                                       num_groups);
  auto d_ptr = allocate_aligned_buffer<typename Gemm::EpilogueOutputOp::ElementOutput*>(
      int_cursor, int_remaining, num_groups);
  auto sfa_ptr = allocate_aligned_buffer<const ElementAccumulator*>(
      int_cursor, int_remaining, num_groups);
  auto sfb_ptr = allocate_aligned_buffer<const ElementAccumulator*>(
      int_cursor, int_remaining, num_groups);
  auto stride_a = allocate_aligned_buffer<StrideA>(int_cursor, int_remaining, num_groups);
  auto stride_b = allocate_aligned_buffer<StrideB>(int_cursor, int_remaining, num_groups);
  auto stride_c = allocate_aligned_buffer<StrideC>(int_cursor, int_remaining, num_groups);
  auto stride_d = allocate_aligned_buffer<StrideD>(int_cursor, int_remaining, num_groups);
  auto layout_sfa = allocate_aligned_buffer<LayoutSFA>(int_cursor, int_remaining, num_groups);
  auto layout_sfb = allocate_aligned_buffer<LayoutSFB>(int_cursor, int_remaining, num_groups);
  if (problem_sizes == nullptr || a_ptr == nullptr || b_ptr == nullptr || c_ptr == nullptr ||
      d_ptr == nullptr || sfa_ptr == nullptr || sfb_ptr == nullptr || stride_a == nullptr ||
      stride_b == nullptr || stride_c == nullptr || stride_d == nullptr ||
      layout_sfa == nullptr || layout_sfb == nullptr) {
    return cudaErrorMemoryAllocation;
  }

  if (m_indptr_host != nullptr && num_groups <= 64) {
    constexpr int kMaxGroups = 64;
    std::array<typename ProblemShape::UnderlyingProblemShape, kMaxGroups> problem_sizes_host_storage{};
    std::array<const typename Gemm::ElementA*, kMaxGroups> a_ptr_host{};
    std::array<const typename Gemm::ElementB*, kMaxGroups> b_ptr_host{};
    std::array<const typename Gemm::ElementC*, kMaxGroups> c_ptr_host{};
    std::array<typename Gemm::EpilogueOutputOp::ElementOutput*, kMaxGroups> d_ptr_host{};
    std::array<const ElementAccumulator*, kMaxGroups> sfa_ptr_host{};
    std::array<const ElementAccumulator*, kMaxGroups> sfb_ptr_host{};
    std::array<StrideA, kMaxGroups> stride_a_host{};
    std::array<StrideB, kMaxGroups> stride_b_host{};
    std::array<StrideC, kMaxGroups> stride_c_host{};
    std::array<StrideD, kMaxGroups> stride_d_host{};
    std::array<LayoutSFA, kMaxGroups> layout_sfa_host{};
    std::array<LayoutSFB, kMaxGroups> layout_sfb_host{};
    int sf_n = n / 128;
    int sf_k = k / 128;
    for (int group = 0; group < num_groups; ++group) {
      int expert_id = group_ids_host == nullptr ? group : group_ids_host[group];
      int m_offset = m_indptr_host[group];
      int m = m_indptr_host[group + 1] - m_offset;
      int sf_m_offset = m_offset / ScaleGranularityM;
      problem_sizes_host_storage[group] = typename ProblemShape::UnderlyingProblemShape(m, n, k);
      a_ptr_host[group] = a_ptr_base + static_cast<int64_t>(m_offset) * k;
      b_ptr_host[group] = b_ptr_base + static_cast<int64_t>(expert_id) * n * k;
      c_ptr_host[group] = d_ptr_base + static_cast<int64_t>(m_offset) * n;
      d_ptr_host[group] = d_ptr_base + static_cast<int64_t>(m_offset) * n;
      sfb_ptr_host[group] = sfb_ptr_base + static_cast<int64_t>(expert_id) * sf_n * sf_k;
      stride_a_host[group] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
      stride_b_host[group] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
      stride_c_host[group] =
          cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
      stride_d_host[group] =
          cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
      if constexpr (ScaleMajorK) {
        layout_sfa_host[group] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
        sfa_ptr_host[group] = sfa_ptr_base + static_cast<int64_t>(sf_m_offset) * sf_k;
      } else {
        layout_sfa_host[group] =
            ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(max_m, n, k, 1));
        sfa_ptr_host[group] = sfa_ptr_base + sf_m_offset;
      }
      layout_sfb_host[group] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    }

    cudaMemcpyAsync(problem_sizes, problem_sizes_host_storage.data(),
                    sizeof(problem_sizes_host_storage[0]) * num_groups, cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(a_ptr, a_ptr_host.data(), sizeof(a_ptr_host[0]) * num_groups,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b_ptr, b_ptr_host.data(), sizeof(b_ptr_host[0]) * num_groups,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(c_ptr, c_ptr_host.data(), sizeof(c_ptr_host[0]) * num_groups,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ptr, d_ptr_host.data(), sizeof(d_ptr_host[0]) * num_groups,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(sfa_ptr, sfa_ptr_host.data(), sizeof(sfa_ptr_host[0]) * num_groups,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(sfb_ptr, sfb_ptr_host.data(), sizeof(sfb_ptr_host[0]) * num_groups,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(stride_a, stride_a_host.data(), sizeof(stride_a_host[0]) * num_groups,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(stride_b, stride_b_host.data(), sizeof(stride_b_host[0]) * num_groups,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(stride_c, stride_c_host.data(), sizeof(stride_c_host[0]) * num_groups,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(stride_d, stride_d_host.data(), sizeof(stride_d_host[0]) * num_groups,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(layout_sfa, layout_sfa_host.data(), sizeof(layout_sfa_host[0]) * num_groups,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(layout_sfb, layout_sfb_host.data(), sizeof(layout_sfb_host[0]) * num_groups,
                    cudaMemcpyHostToDevice, stream);
  } else {
    int threads = num_groups < 1024 ? num_groups : 1024;
    int blocks = (num_groups + threads - 1) / threads;
    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = true;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto prepare_args_kernel =
        compute_sm1xx_cutlass_group_gemm_args<ScaleConfig, ElementA, float, typename Gemm::ElementC,
                                              ElementD,
                                              typename ProblemShape::UnderlyingProblemShape,
                                              StrideA, StrideB, StrideC, StrideD, LayoutSFA,
                                              LayoutSFB, ScaleMajorK>;
    cudaError_t launch_status = cudaLaunchKernelEx(
        &config, prepare_args_kernel, a_ptr_base, b_ptr_base, sfa_ptr_base, sfb_ptr_base,
        d_ptr_base, m_indptr, max_m, n, k, num_groups, ScaleGranularityM, 128, 128,
        problem_sizes, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, d_ptr, stride_a, stride_b,
        stride_c, stride_d, layout_sfa, layout_sfb);
    if (launch_status != cudaSuccess) {
      return launch_status;
    }
  }

  thread_local int const sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;
  std::array<typename ProblemShape::UnderlyingProblemShape, 64> problem_sizes_host_storage;
  typename ProblemShape::UnderlyingProblemShape const* problem_sizes_host = nullptr;
  if (m_indptr_host != nullptr && num_groups <= static_cast<int>(problem_sizes_host_storage.size())) {
    for (int group = 0; group < num_groups; ++group) {
      int m = m_indptr_host[group + 1] - m_indptr_host[group];
      problem_sizes_host_storage[group] = typename ProblemShape::UnderlyingProblemShape(m, n, k);
    }
    problem_sizes_host = problem_sizes_host_storage.data();
  }
  typename Gemm::Arguments arguments;
  arguments.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
  arguments.problem_shape = {num_groups, problem_sizes, problem_sizes_host};
  arguments.mainloop = {a_ptr, stride_a, b_ptr, stride_b, sfa_ptr, layout_sfa,
                        sfb_ptr, layout_sfb};
  arguments.epilogue = {{}, c_ptr, stride_c, d_ptr, stride_d};
  arguments.hw_info = hw_info;
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  Gemm gemm;
  size_t required_workspace = Gemm::get_workspace_size(arguments);
  char* workspace_cursor = reinterpret_cast<char*>(workspace);
  size_t workspace_remaining = workspace_size_in_bytes;
  void* workspace_ptr = allocate_aligned_bytes(workspace_cursor, workspace_remaining,
                                               required_workspace);
  if (workspace_ptr == nullptr) {
    return cudaErrorMemoryAllocation;
  }
  cudaError_t status = cutlass_status_to_cuda(gemm.can_implement(arguments));
  if (status != cudaSuccess) return status;
  status = cutlass_status_to_cuda(gemm.initialize(arguments, workspace_ptr));
  if (status != cudaSuccess) return status;
  // Keep this launch stream-ordered: the MoE pipeline immediately consumes the
  // grouped GEMM output in SwiGLU/GEMM2 kernels on the same stream.
  return cutlass_status_to_cuda(gemm.run(stream));
}

template <int ScaleGranularityM, bool ScaleMajorK, int MmaTileM, int MmaSM, typename DTypeA,
          typename DTypeB, typename DTypeOut, int MmaTileN = 128>
cudaError_t launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_select_epilogue(
    bool use_tma_epilogue, void* arg_buffer, size_t arg_buffer_size_in_bytes, void* workspace,
    size_t workspace_size_in_bytes, DTypeA* a_ptr_base, DTypeB* b_ptr_base,
    float* sfa_ptr_base, float* sfb_ptr_base, DTypeOut* d_ptr_base, int* m_indptr,
    int max_m, int n, int k, int num_groups, cudaStream_t stream,
    const int* m_indptr_host = nullptr, const int* group_ids_host = nullptr) {
  if (use_tma_epilogue) {
    return launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
        ScaleGranularityM, ScaleMajorK, MmaTileM, MmaSM, DTypeA, DTypeB, DTypeOut, true,
        MmaTileN>(
        arg_buffer, arg_buffer_size_in_bytes, workspace, workspace_size_in_bytes, a_ptr_base,
        b_ptr_base, sfa_ptr_base, sfb_ptr_base, d_ptr_base, m_indptr, max_m, n, k, num_groups,
        stream, m_indptr_host, group_ids_host);
  }
  return launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
      ScaleGranularityM, ScaleMajorK, MmaTileM, MmaSM, DTypeA, DTypeB, DTypeOut, false,
      MmaTileN>(
      arg_buffer, arg_buffer_size_in_bytes, workspace, workspace_size_in_bytes, a_ptr_base,
      b_ptr_base, sfa_ptr_base, sfb_ptr_base, d_ptr_base, m_indptr, max_m, n, k, num_groups,
      stream, m_indptr_host, group_ids_host);
}

template <int MmaTileM, int MmaSM, typename DTypeA, typename DTypeB, typename DTypeOut>
cudaError_t launch_cutlass_dense_gemm_sm100(void* workspace, size_t workspace_size_in_bytes,
                                            DTypeA* a_ptr, DTypeB* b_ptr, DTypeOut* d_ptr,
                                            int m, int n, int k, cudaStream_t stream) {
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

  using MmaTileShape = Shape<Int<MmaTileM>, _128, _64>;
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

template <typename DTypeA, typename DTypeB, typename DTypeC, typename DTypeOut,
          typename ProblemShape, typename StrideA, typename StrideB, typename StrideC,
          typename StrideD>
__global__ void compute_dense_group_gemm_args(
    DTypeA* a, DTypeB* b, DTypeOut* d, int* m_indptr, int n, int k, int num_groups,
    ProblemShape* problem_sizes, const DTypeA** a_ptr, const DTypeB** b_ptr,
    const DTypeC** c_ptr, DTypeOut** d_ptr, StrideA* stride_a, StrideB* stride_b,
    StrideC* stride_c, StrideD* stride_d) {
  int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= num_groups) return;
  int m_offset = m_indptr[group];
  int m = m_indptr[group + 1] - m_offset;
  problem_sizes[group] = ProblemShape(m, n, k);
  stride_a[group] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  stride_b[group] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  // Same ptr-array grouped epilogue contract as blockscaled grouped GEMM.
  stride_c[group] = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  stride_d[group] = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
  a_ptr[group] = a + static_cast<int64_t>(m_offset) * k;
  b_ptr[group] = b + static_cast<int64_t>(group) * n * k;
  c_ptr[group] = reinterpret_cast<const DTypeC*>(d + static_cast<int64_t>(m_offset) * n);
  d_ptr[group] = d + static_cast<int64_t>(m_offset) * n;
}

template <int MmaTileM, int MmaSM, typename DTypeA, typename DTypeB, typename DTypeOut>
cudaError_t launch_cutlass_dense_grouped_ptr_gemm_sm100(
    void* arg_buffer, size_t arg_buffer_size_in_bytes, void* workspace,
    size_t workspace_size_in_bytes, DTypeA* a_base, DTypeB* b_base, DTypeOut* d_base,
    int* m_indptr, int n, int k, int num_groups, cudaStream_t stream,
    const int* m_indptr_host = nullptr) {
  using namespace cute;
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using ElementA = DTypeA;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  using ElementB = DTypeB;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  using ElementD = DTypeOut;
  using LayoutD = cutlass::layout::RowMajor;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ElementC = DTypeOut;
  using LayoutC = LayoutD;
  constexpr int AlignmentC = AlignmentD;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using MmaTileShape = Shape<Int<MmaTileM>, _128, _64>;
  using ClusterShape = Shape<Int<MmaSM>, _1, _1>;
  using EpilogueSchedule =
      std::conditional_t<MmaSM == 1, cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm,
                         cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, LayoutC*, AlignmentC, ElementD, LayoutD*, AlignmentD,
      EpilogueSchedule>::CollectiveOp;
  using MainloopSchedule =
      std::conditional_t<MmaSM == 1, cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100,
                         cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100>;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA, LayoutA*, AlignmentA,
      ElementB, LayoutB*, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  char* cursor = reinterpret_cast<char*>(arg_buffer);
  size_t remaining = arg_buffer_size_in_bytes;
  auto problem_sizes = allocate_aligned_buffer<typename ProblemShape::UnderlyingProblemShape>(
      cursor, remaining, num_groups);
  auto a_ptr = allocate_aligned_buffer<const typename Gemm::ElementA*>(cursor, remaining,
                                                                       num_groups);
  auto b_ptr = allocate_aligned_buffer<const typename Gemm::ElementB*>(cursor, remaining,
                                                                       num_groups);
  auto c_ptr = allocate_aligned_buffer<const typename Gemm::ElementC*>(cursor, remaining,
                                                                       num_groups);
  auto d_ptr = allocate_aligned_buffer<typename Gemm::EpilogueOutputOp::ElementOutput*>(
      cursor, remaining, num_groups);
  auto stride_a = allocate_aligned_buffer<StrideA>(cursor, remaining, num_groups);
  auto stride_b = allocate_aligned_buffer<StrideB>(cursor, remaining, num_groups);
  auto stride_c = allocate_aligned_buffer<StrideC>(cursor, remaining, num_groups);
  auto stride_d = allocate_aligned_buffer<StrideD>(cursor, remaining, num_groups);
  if (problem_sizes == nullptr || a_ptr == nullptr || b_ptr == nullptr || c_ptr == nullptr ||
      d_ptr == nullptr || stride_a == nullptr || stride_b == nullptr || stride_c == nullptr ||
      stride_d == nullptr) {
    return cudaErrorMemoryAllocation;
  }

  if (m_indptr_host != nullptr && num_groups <= 64) {
    constexpr int kMaxGroups = 64;
    std::array<typename ProblemShape::UnderlyingProblemShape, kMaxGroups> problem_sizes_host{};
    std::array<const typename Gemm::ElementA*, kMaxGroups> a_ptr_host{};
    std::array<const typename Gemm::ElementB*, kMaxGroups> b_ptr_host{};
    std::array<const typename Gemm::ElementC*, kMaxGroups> c_ptr_host{};
    std::array<typename Gemm::EpilogueOutputOp::ElementOutput*, kMaxGroups> d_ptr_host{};
    std::array<StrideA, kMaxGroups> stride_a_host{};
    std::array<StrideB, kMaxGroups> stride_b_host{};
    std::array<StrideC, kMaxGroups> stride_c_host{};
    std::array<StrideD, kMaxGroups> stride_d_host{};
    for (int group = 0; group < num_groups; ++group) {
      int m_offset = m_indptr_host[group];
      int m = m_indptr_host[group + 1] - m_offset;
      problem_sizes_host[group] = typename ProblemShape::UnderlyingProblemShape(m, n, k);
      a_ptr_host[group] = a_base + static_cast<int64_t>(m_offset) * k;
      b_ptr_host[group] = b_base + static_cast<int64_t>(group) * n * k;
      c_ptr_host[group] = d_base + static_cast<int64_t>(m_offset) * n;
      d_ptr_host[group] = d_base + static_cast<int64_t>(m_offset) * n;
      stride_a_host[group] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
      stride_b_host[group] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
      stride_c_host[group] =
          cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
      stride_d_host[group] =
          cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
    }
    cudaMemcpy(problem_sizes, problem_sizes_host.data(), sizeof(problem_sizes_host[0]) * num_groups,
               cudaMemcpyHostToDevice);
    cudaMemcpy(a_ptr, a_ptr_host.data(), sizeof(a_ptr_host[0]) * num_groups,
               cudaMemcpyHostToDevice);
    cudaMemcpy(b_ptr, b_ptr_host.data(), sizeof(b_ptr_host[0]) * num_groups,
               cudaMemcpyHostToDevice);
    cudaMemcpy(c_ptr, c_ptr_host.data(), sizeof(c_ptr_host[0]) * num_groups,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr, d_ptr_host.data(), sizeof(d_ptr_host[0]) * num_groups,
               cudaMemcpyHostToDevice);
    cudaMemcpy(stride_a, stride_a_host.data(), sizeof(stride_a_host[0]) * num_groups,
               cudaMemcpyHostToDevice);
    cudaMemcpy(stride_b, stride_b_host.data(), sizeof(stride_b_host[0]) * num_groups,
               cudaMemcpyHostToDevice);
    cudaMemcpy(stride_c, stride_c_host.data(), sizeof(stride_c_host[0]) * num_groups,
               cudaMemcpyHostToDevice);
    cudaMemcpy(stride_d, stride_d_host.data(), sizeof(stride_d_host[0]) * num_groups,
               cudaMemcpyHostToDevice);
  } else {
    int threads = num_groups < 1024 ? num_groups : 1024;
    int blocks = (num_groups + threads - 1) / threads;
    compute_dense_group_gemm_args<ElementA, ElementB, typename Gemm::ElementC, ElementD,
                                  typename ProblemShape::UnderlyingProblemShape, StrideA, StrideB,
                                  StrideC, StrideD><<<blocks, threads, 0, stream>>>(
        a_base, b_base, d_base, m_indptr, n, k, num_groups, problem_sizes, a_ptr, b_ptr,
        c_ptr, d_ptr, stride_a, stride_b, stride_c, stride_d);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) return launch_status;
  }

  thread_local int const sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;
  std::array<typename ProblemShape::UnderlyingProblemShape, 64> problem_sizes_host_storage{};
  typename ProblemShape::UnderlyingProblemShape const* problem_sizes_host = nullptr;
  if (m_indptr_host != nullptr && num_groups <= static_cast<int>(problem_sizes_host_storage.size())) {
    for (int group = 0; group < num_groups; ++group) {
      int m = m_indptr_host[group + 1] - m_indptr_host[group];
      problem_sizes_host_storage[group] = typename ProblemShape::UnderlyingProblemShape(m, n, k);
    }
    problem_sizes_host = problem_sizes_host_storage.data();
  }
  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                     {num_groups, problem_sizes, problem_sizes_host},
                                     {a_ptr, stride_a, b_ptr, stride_b},
                                     {{}, c_ptr, stride_c, d_ptr, stride_d},
                                     hw_info};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  Gemm gemm;
  size_t required_workspace = Gemm::get_workspace_size(arguments);
  char* workspace_cursor = reinterpret_cast<char*>(workspace);
  size_t workspace_remaining = workspace_size_in_bytes;
  void* workspace_ptr = allocate_aligned_bytes(workspace_cursor, workspace_remaining,
                                               required_workspace);
  if (workspace_ptr == nullptr) return cudaErrorMemoryAllocation;
  cudaError_t status = cutlass_status_to_cuda(gemm.can_implement(arguments));
  if (status != cudaSuccess) return status;
  status = cutlass_status_to_cuda(gemm.initialize(arguments, workspace_ptr));
  if (status != cudaSuccess) return status;
  return cutlass_status_to_cuda(gemm.run(stream));
}

#if FIB_HAS_FLASHINFER_GROUP_GEMM_FP8_SM100
template <bool ScaleMajorK, int MmaSM, typename DTypeOut>
cudaError_t launch_flashinfer_grouped_blockscaled_gemm1_sm100(
    void* arg_buffer, size_t arg_buffer_size_in_bytes, void* workspace,
    size_t workspace_size_in_bytes, cutlass::float_e4m3_t* a_ptr,
    cutlass::float_e4m3_t* b_ptr, float* sfa_ptr, float* sfb_ptr, DTypeOut* d_ptr,
    int* m_indptr, int max_m, int n, int k, int num_groups, cudaStream_t stream) {
  return flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM100<
      1, 128, 128, ScaleMajorK, MmaSM>(
      arg_buffer, arg_buffer_size_in_bytes, workspace, workspace_size_in_bytes, a_ptr, b_ptr,
      sfa_ptr, sfb_ptr, d_ptr, m_indptr, max_m, n, k, num_groups, stream);
}
#endif

#endif  // FIB_HAS_DIRECT_CUTLASS_SM100

}  // namespace mxfp::detail
