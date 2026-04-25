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
#include "mxfp_cutlass_sm100_g1_paired_collective.cuh"
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

template <typename BaseCollective>
struct G2InputQuantCollective : BaseCollective {
  using Base = BaseCollective;
  using ClusterShape = typename Base::DispatchPolicy::ClusterShape;

  struct Arguments {
    typename Base::Arguments base{};
    float const** ptr_activation{nullptr};
  };

  struct Params {
    typename Base::Params base{};
    float const** ptr_activation{nullptr};
  };

  CUTLASS_DEVICE
  G2InputQuantCollective(Params const& params, ClusterShape cluster_shape,
                         uint32_t block_rank_in_cluster)
      : Base(params.base, cluster_shape, block_rank_in_cluster),
        ptr_activation_(params.ptr_activation) {}

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape problem_shapes, Arguments const& args, void* workspace,
      cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {
    return {Base::to_underlying_arguments(problem_shapes, args.base, workspace, hw_info),
            args.ptr_activation};
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args,
                                   int sm_count) {
    return Base::get_workspace_size(problem_shape, args.base, sm_count);
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const& problem_shape,
                                              Arguments const& args, void* workspace,
                                              cudaStream_t stream,
                                              cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    return Base::initialize_workspace(problem_shape, args.base, workspace, stream, cuda_adapter);
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape problem_shapes, Arguments const& args) {
    return Base::can_implement(problem_shapes, args.base);
  }

  template <class ProblemShape_MNKL, class SharedTensors, class TensorMapStorage>
  CUTLASS_DEVICE auto load_ab_init(ProblemShape_MNKL const& problem_shape_mnkl,
                                   Params const& params, SharedTensors& shared_tensors,
                                   TensorMapStorage& shared_tensormaps, int32_t sm_count,
                                   int32_t sm_idx, int32_t init_group) const {
    return Base::load_ab_init(problem_shape_mnkl, params.base, shared_tensors, shared_tensormaps,
                              sm_count, sm_idx, init_group);
  }

  template <class ProblemShape_MNKL, class SharedTensors>
  CUTLASS_DEVICE auto load_sf_init(ProblemShape_MNKL const& problem_shape_mnkl,
                                   Params const& params, SharedTensors& shared_tensors,
                                   int current_group) const {
    return Base::load_sf_init(problem_shape_mnkl, params.base, shared_tensors, current_group);
  }

  template <class ProblemShape_MNKL, class SharedTensors>
  CUTLASS_DEVICE auto load_sf_update(ProblemShape_MNKL const& problem_shape_mnkl,
                                     Params const& params, SharedTensors& shared_tensors,
                                     int current_group) const {
    return Base::load_sf_update(problem_shape_mnkl, params.base, shared_tensors, current_group);
  }

  template <class... Ts>
  CUTLASS_DEVICE auto mma_init(Params const& params, Ts&&... args) const {
    return Base::mma_init(params.base, static_cast<Ts&&>(args)...);
  }

  template <class... Ts>
  CUTLASS_DEVICE auto load_ab(Params const& params, Ts&&... args) {
    return Base::load_ab(params.base, static_cast<Ts&&>(args)...);
  }

  template <class TensorMapStorage, class TensorMapA, class TensorMapB, class ProblemShape>
  CUTLASS_DEVICE void tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps, Params const& params,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps, ProblemShape problem_shape,
      int32_t next_batch) {
    Base::tensormaps_perform_update(shared_tensormaps, params.base, input_tensormaps,
                                    problem_shape, next_batch);
  }

  float const** ptr_activation_{nullptr};
};

template <typename BaseEpilogue>
struct G1SwiGLUFusedNoGmemEpilogue : BaseEpilogue {
  using Base = BaseEpilogue;
  using SharedStorage = typename Base::SharedStorage;
  using TensorStorage = typename Base::TensorStorage;
  using TensorMapStorage = typename Base::TensorMapStorage;

  struct Arguments {
    typename Base::Arguments base{};
    uint8_t** ptr_activation{nullptr};
    float** ptr_activation_scale{nullptr};
    int intermediate{0};
    int pair_offset{16};
    int mode{0};
  };

  struct Params {
    typename Base::Params base{};
    uint8_t** ptr_activation{nullptr};
    float** ptr_activation_scale{nullptr};
    int intermediate{0};
    int pair_offset{16};
    int mode{0};
  };

  CUTLASS_HOST_DEVICE
  G1SwiGLUFusedNoGmemEpilogue(Params const& params, SharedStorage& shared_tensors)
      : Base(params.base, shared_tensors),
        ptr_activation_(params.ptr_activation),
        ptr_activation_scale_(params.ptr_activation_scale),
        intermediate_(params.intermediate),
        pair_offset_(params.pair_offset),
        mode_(params.mode) {}

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape,
                                                  Arguments const& args, void* workspace) {
    return {Base::to_underlying_arguments(problem_shape, args.base, workspace),
            args.ptr_activation, args.ptr_activation_scale, args.intermediate, args.pair_offset,
            args.mode};
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args,
                                   int sm_count = 0) {
    return Base::get_workspace_size(problem_shape, args.base, sm_count);
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const& problem_shape,
                                              Arguments const& args, void* workspace,
                                              cudaStream_t stream,
                                              cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    return Base::initialize_workspace(problem_shape, args.base, workspace, stream, cuda_adapter);
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return Base::can_implement(problem_shape, args.base);
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    Base::prefetch_tma_descriptors(params.base);
  }

  template <bool... Args>
  CUTLASS_DEVICE auto tensormaps_init(Params const& params, TensorMapStorage& shared_tensormaps,
                                      int32_t sm_count, int32_t sm_idx,
                                      int32_t warp_group_idx = 0) const {
    return Base::template tensormaps_init<Args...>(params.base, shared_tensormaps, sm_count,
                                                   sm_idx, warp_group_idx);
  }

  template <bool... Args>
  CUTLASS_DEVICE auto load_init(Params const& params, TensorMapStorage& shared_tensormap,
                                int32_t sm_count, int32_t sm_idx) const {
    return Base::template load_init<Args...>(params.base, shared_tensormap, sm_count, sm_idx);
  }

  template <bool... Args>
  CUTLASS_DEVICE auto store_init(Params const& params, TensorMapStorage& shared_tensormap,
                                 int32_t sm_count, int32_t sm_idx) const {
    return Base::template store_init<Args...>(params.base, shared_tensormap, sm_count, sm_idx);
  }

  template <bool IsLoad, bool WaitForInflightTmaRequests = true, class ProblemShape>
  CUTLASS_DEVICE void tensormaps_perform_update(TensorMapStorage& shared_tensormap,
                                                Params const& params,
                                                cute::TmaDescriptor const* tensormap,
                                                ProblemShape problem_shape,
                                                int32_t next_batch) {
    Base::template tensormaps_perform_update<IsLoad, WaitForInflightTmaRequests>(
        shared_tensormap, params.base, tensormap, problem_shape, next_batch);
  }

  CUTLASS_DEVICE static float silu(float x) { return x / (1.0f + __expf(-x)); }

  CUTLASS_DEVICE static uint8_t float_to_e4m3(float x) {
    __nv_fp8_e4m3 y(x);
    return *reinterpret_cast<uint8_t*>(&y);
  }

  template <class CtaTileMNK>
  CUTLASS_DEVICE void store_tail(typename Base::LoadPipeline,
                                 typename Base::LoadPipelineState,
                                 typename Base::StorePipeline,
                                 typename Base::StorePipelineState,
                                 CtaTileMNK) {
    // This POC bypasses the base TMA D-store path and writes compact FP8
    // activation directly, so there are no store-pipeline commits to drain.
  }

  template <bool ReuseTmem = false, bool WaitForInflightTmaRequests = true,
            class AccumulatorPipeline, class AccumulatorPipelineState, class ProblemShapeMNKL,
            class CtaTileMNK, class TileCoordMNKL, class MmaTileMNK, class TiledMma,
            class AccEngine, class AccLayout>
  CUTLASS_DEVICE auto store(typename Base::LoadPipeline load_pipeline,
                            typename Base::LoadPipelineState load_pipe_consumer_state,
                            typename Base::StorePipeline store_pipeline,
                            typename Base::StorePipelineState store_pipe_producer_state,
                            AccumulatorPipeline acc_pipeline,
                            AccumulatorPipelineState acc_pipe_consumer_state,
                            ProblemShapeMNKL problem_shape_mnkl, CtaTileMNK cta_tile_mnk,
                            TileCoordMNKL cta_coord_mnkl, MmaTileMNK mma_tile_mnk,
                            TiledMma tiled_mma, cute::Tensor<AccEngine, AccLayout> accumulators,
                            TensorStorage& shared_tensors) {
    using namespace cute;
    (void)load_pipeline;
    (void)store_pipeline;
    (void)mma_tile_mnk;
    (void)tiled_mma;
    (void)shared_tensors;
    (void)WaitForInflightTmaRequests;

    using ElementAccumulator = typename AccEngine::value_type;
    using CopyOpT2R = typename Base::CopyOpT2R;
    constexpr int ThreadCount = Base::ThreadCount;
    constexpr int FragmentSize = size(typename Base::EpilogueTile{}) / ThreadCount;
    static_assert(FragmentSize == 128,
                  "G1 SwiGLU fused epilogue assumes one 128-column half per thread.");

    acc_pipeline.consumer_wait(acc_pipe_consumer_state);

    auto [M, N, K, L] = problem_shape_mnkl;
    (void)K;
    auto problem_shape_mnl = select<0, 1, 3>(problem_shape_mnkl);
    auto cta_coord_mnl = select<0, 1, 3>(cta_coord_mnkl);
    auto cta_tiler = take<0, 2>(cta_tile_mnk);
    int thread_idx = threadIdx.x % ThreadCount;

    Tensor tAcc = accumulators(make_coord(_, _), _0{}, _0{});
    Tensor tAcc_epi = flat_divide(tAcc, typename Base::EpilogueTile{});
    TiledCopy tiled_t2r = make_tmem_copy(CopyOpT2R{}, tAcc_epi(_, _, _0{}, _0{}));
    ThrCopy thread_t2r = tiled_t2r.get_slice(thread_idx);
    Tensor tTR_tAcc = thread_t2r.partition_S(tAcc_epi);

    Tensor coordD = make_identity_tensor(problem_shape_mnl);
    Tensor cD = local_tile(coordD, cta_tiler, cta_coord_mnl);
    Tensor cD_epi = flat_divide(cD, typename Base::EpilogueTile{});
    Tensor tTR_cD = thread_t2r.partition_D(cD_epi);
    Tensor tTR_rAcc = make_tensor<ElementAccumulator>(shape(tTR_cD(_, _, _, _0{}, _0{})));
    Tensor tTR_rAcc_frg =
        recast<cutlass::Array<ElementAccumulator, FragmentSize>>(coalesce(tTR_rAcc));

    constexpr int NumEpiSubtilesN = CUTE_STATIC_V(size<4>(tTR_tAcc));
    constexpr int NumEpiSubtilesM = CUTE_STATIC_V(size<3>(tTR_tAcc));
    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < NumEpiSubtilesN; ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < NumEpiSubtilesM; ++epi_m) {
        Tensor tTR_tAcc_mn = tTR_tAcc(_, _, _, epi_m, epi_n);
        copy(tiled_t2r, tTR_tAcc_mn, tTR_rAcc);

        bool is_last_iteration = epi_m == size<3>(tTR_tAcc) - 1 &&
                                 epi_n == size<4>(tTR_tAcc) - 1;
        if (is_last_iteration) {
          cutlass::arch::fence_view_async_tmem_load();
          acc_pipeline.consumer_release(acc_pipe_consumer_state);
          ++acc_pipe_consumer_state;
        }

        Tensor coords = coalesce(tTR_cD(_, _, _, epi_m, epi_n));
        auto c0 = coords(_0{});
        int row = int(get<0>(c0));
        int full_n0 = int(get<1>(c0));
        int group = int(get<2>(c0));
        int lane = thread_idx & 31;
        int paired_lane = (lane < 16) ? lane + 16 : lane - 16;
        bool gate_thread = ((full_n0 & 128) == 0) && lane < 16;
        bool in_bounds = row < int(M) && full_n0 < int(N) && group < int(L);

        auto acc_arr = tTR_rAcc_frg(_0{});
        float vals[FragmentSize];
        float max_abs = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          float gate = acc_arr[i];
          float up = __shfl_sync(0xffffffffu, gate, paired_lane);
          float v = gate * silu(up);
          vals[i] = v;
          max_abs = fmaxf(max_abs, fabsf(v));
        }

        if (gate_thread && in_bounds) {
          int out_block = full_n0 >> 8;
          int out_col0 = out_block * FragmentSize;
          float scale = fmaxf(max_abs / 448.0f, 1.0e-8f);
          uint8_t* out = ptr_activation_[group] + static_cast<int64_t>(row) * intermediate_;
          float* out_scale = ptr_activation_scale_[group] +
                             static_cast<int64_t>(row) * (intermediate_ / FragmentSize);
          out_scale[out_block] = scale;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < FragmentSize; ++i) {
            int col = out_col0 + i;
            if (col < intermediate_) out[col] = float_to_e4m3(vals[i] / scale);
          }
        }
      }
    }

    ++store_pipe_producer_state;
    return make_tuple(load_pipe_consumer_state, store_pipe_producer_state,
                      acc_pipe_consumer_state);
  }

  template <class ProblemShapeMNKL, class CtaTileMNK, class TileCoordMNKL,
            class MmaTileMNK, class TiledMma, class AccEngine, class AccLayout,
            class TensorMapD, class TiledCopyT2R>
  CUTLASS_DEVICE auto store(typename Base::LoadPipeline load_pipeline,
                            typename Base::LoadPipelineState load_pipe_consumer_state,
                            typename Base::StorePipeline store_pipeline,
                            typename Base::StorePipelineState store_pipe_producer_state,
                            ProblemShapeMNKL problem_shape_mnkl, CtaTileMNK cta_tile_mnk,
                            TileCoordMNKL cta_coord_mnkl, MmaTileMNK mma_tile_mnk,
                            TiledMma tiled_mma,
                            cute::Tensor<AccEngine, AccLayout>& tTR_rAcc,
                            TensorStorage& shared_tensors, TensorMapD store_tensormap,
                            TiledCopyT2R tiled_t2r) {
    using namespace cute;
    (void)load_pipeline;
    (void)store_pipeline;
    (void)mma_tile_mnk;
    (void)tiled_mma;
    (void)shared_tensors;
    (void)store_tensormap;

    using ElementAccumulator = typename AccEngine::value_type;
    constexpr int ThreadCount = Base::ThreadCount;
    constexpr int FragmentSize = 128;
    static_assert(FragmentSize == 128,
                  "G1 SwiGLU fused epilogue assumes one 128-column half per thread.");
    static_assert(is_rmem<AccEngine>::value, "Accumulator must be register resident.");
    static_assert(rank(AccLayout{}) == 5,
                  "Accumulator must be copy-partitioned: (T2R,T2R_M,T2R_N,EPI_M,EPI_N).");

    auto [M, N, K, L] = problem_shape_mnkl;
    (void)K;
    (void)L;
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;
    (void)k_coord;
    int group = int(l_coord);
    int thread_idx = threadIdx.x % size(tiled_t2r);
    auto thread_t2r = tiled_t2r.get_slice(thread_idx);

    Tensor coordD = make_identity_tensor(make_shape(M, N));
    Tensor cD = local_tile(coordD, take<0, 2>(cta_tile_mnk), make_coord(m_coord, n_coord));
    Tensor tTR_cD = thread_t2r.partition_D(flat_divide(cD, typename Base::EpilogueTile{}));

    constexpr int NumEpiSubtilesN = CUTE_STATIC_V(size<4>(tTR_rAcc));
    constexpr int NumEpiSubtilesM = CUTE_STATIC_V(size<3>(tTR_rAcc));
    constexpr int CtaM = CUTE_STATIC_V(size<0>(CtaTileMNK{}));
    constexpr int CtaN = CUTE_STATIC_V(size<1>(CtaTileMNK{}));
    static_assert(CtaM == 64 && CtaN == 256,
                  "Shared-tile G1 fused POC currently supports CTA 64x256 only.");

    if (mode_ == 1) {
      int lane = threadIdx.x & 31;
      int pair_offset = pair_offset_;
      if (pair_offset <= 0 || pair_offset > 16) pair_offset = 16;
      int paired_lane = (lane < pair_offset) ? lane + pair_offset : lane - pair_offset;

      CUTLASS_PRAGMA_UNROLL
      for (int epi_n = 0; epi_n < NumEpiSubtilesN; ++epi_n) {
        CUTLASS_PRAGMA_UNROLL
        for (int epi_m = 0; epi_m < NumEpiSubtilesM; ++epi_m) {
          Tensor tTR_rAcc_tile = tTR_rAcc(_, _, _, epi_m, epi_n);
          Tensor tTR_rAcc_flat = coalesce(tTR_rAcc_tile);
          static_assert(CUTE_STATIC_V(size(tTR_rAcc_flat)) == FragmentSize,
                        "Unexpected G1 fused register fragment size.");
          Tensor coords = coalesce(tTR_cD(_, _, _, epi_m, epi_n));
          auto c0 = coords(_0{});
          int row0 = int(get<0>(c0));
          int full_n0 = int(get<1>(c0));
          int tile_col0 = int(n_coord) * CtaN;
          int local_n0 = full_n0 - tile_col0;
          int out_block = full_n0 >> 8;
          bool gate_thread = row0 < int(M) && full_n0 < int(N) && local_n0 >= 0 &&
                             local_n0 < 128 && lane < pair_offset;

          float vals[FragmentSize];
          float max_abs = 0.0f;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < FragmentSize; ++i) {
            float x = static_cast<float>(tTR_rAcc_flat(i));
            float up = __shfl_sync(0xffffffffu, x, paired_lane);
            float v = x * silu(up);
            vals[i] = v;
            if (gate_thread) max_abs = fmaxf(max_abs, fabsf(v));
          }

          if (gate_thread) {
            float scale = fmaxf(max_abs / 448.0f, 1.0e-8f);
            uint8_t* out =
                ptr_activation_[group] + static_cast<int64_t>(row0) * intermediate_;
            float* out_scale = ptr_activation_scale_[group] +
                               static_cast<int64_t>(row0) * (intermediate_ / 128);
            out_scale[out_block] = scale;
            int col0 = out_block * 128;
            if (col0 + FragmentSize <= intermediate_) {
              uint32_t* out32 = reinterpret_cast<uint32_t*>(out + col0);
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < FragmentSize; i += 4) {
                uint32_t packed =
                    static_cast<uint32_t>(float_to_e4m3(vals[i + 0] / scale)) |
                    (static_cast<uint32_t>(float_to_e4m3(vals[i + 1] / scale)) << 8) |
                    (static_cast<uint32_t>(float_to_e4m3(vals[i + 2] / scale)) << 16) |
                    (static_cast<uint32_t>(float_to_e4m3(vals[i + 3] / scale)) << 24);
                out32[i / 4] = packed;
              }
            } else {
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < FragmentSize; ++i) {
                auto ci = coords(i);
                int full_n = int(get<1>(ci));
                int local_n = full_n - tile_col0;
                int col = out_block * 128 + local_n;
                if (col < intermediate_) out[col] = float_to_e4m3(vals[i] / scale);
              }
            }
          }
        }
      }

      return make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
    }

    __shared__ float up_tile_smem[64][128];
    __shared__ int row_max_bits_smem[64];
    auto synchronize = []() CUTLASS_LAMBDA_FUNC_INLINE {
      cutlass::arch::NamedBarrier::sync(ThreadCount,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    };
    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < NumEpiSubtilesN; ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < NumEpiSubtilesM; ++epi_m) {
        if (threadIdx.x < 64) row_max_bits_smem[threadIdx.x] = 0;

        Tensor tTR_rAcc_tile = tTR_rAcc(_, _, _, epi_m, epi_n);
        Tensor tTR_rAcc_flat = coalesce(tTR_rAcc_tile);
        static_assert(CUTE_STATIC_V(size(tTR_rAcc_flat)) == FragmentSize,
                      "Unexpected G1 fused register fragment size.");
        Tensor coords = coalesce(tTR_cD(_, _, _, epi_m, epi_n));
        auto c0 = coords(_0{});
        int full_n0 = int(get<1>(c0));
        int tile_row0 = int(m_coord) * CtaM;
        int tile_col0 = int(n_coord) * CtaN;
        int out_block = full_n0 >> 8;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          auto ci = coords(i);
          int row = int(get<0>(ci));
          int full_n = int(get<1>(ci));
          int local_row = row - tile_row0;
          int local_n = full_n - tile_col0;
          if (row < int(M) && full_n < int(N) && local_row >= 0 && local_row < CtaM &&
              local_n >= 128 && local_n < CtaN) {
            up_tile_smem[local_row][local_n - 128] = static_cast<float>(tTR_rAcc_flat(i));
          }
        }
        synchronize();

        float vals[FragmentSize];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          auto ci = coords(i);
          int row = int(get<0>(ci));
          int full_n = int(get<1>(ci));
          int local_row = row - tile_row0;
          int local_n = full_n - tile_col0;
          bool valid_gate = row < int(M) && full_n < int(N) && local_row >= 0 &&
                            local_row < CtaM && local_n >= 0 && local_n < 128;
          float v = 0.0f;
          if (valid_gate) {
            float gate = static_cast<float>(tTR_rAcc_flat(i));
            float up = up_tile_smem[local_row][local_n];
            v = gate * silu(up);
            atomicMax(&row_max_bits_smem[local_row], __float_as_int(fabsf(v)));
          }
          vals[i] = v;
        }
        synchronize();

        if (threadIdx.x < 64) {
          int row = tile_row0 + int(threadIdx.x);
          if (row < int(M)) {
            float scale = fmaxf(__int_as_float(row_max_bits_smem[threadIdx.x]) / 448.0f,
                                1.0e-8f);
            float* out_scale = ptr_activation_scale_[group] +
                               static_cast<int64_t>(row) * (intermediate_ / 128);
            out_scale[out_block] = scale;
          }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          auto ci = coords(i);
          int row = int(get<0>(ci));
          int full_n = int(get<1>(ci));
          int local_row = row - tile_row0;
          int local_n = full_n - tile_col0;
          bool valid_gate = row < int(M) && full_n < int(N) && local_row >= 0 &&
                            local_row < CtaM && local_n >= 0 && local_n < 128;
          if (!valid_gate) continue;
          float scale = fmaxf(__int_as_float(row_max_bits_smem[local_row]) / 448.0f,
                              1.0e-8f);
          int col = out_block * 128 + local_n;
          uint8_t* out = ptr_activation_[group] + static_cast<int64_t>(row) * intermediate_;
          if (col < intermediate_) out[col] = float_to_e4m3(vals[i] / scale);
        }
        synchronize();
      }
    }

    return make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  uint8_t** ptr_activation_{nullptr};
  float** ptr_activation_scale_{nullptr};
  int intermediate_{0};
  int pair_offset_{16};
  int mode_{0};
};

template <typename Collective, bool UsePairedBValueTmaRemap = false,
          bool UsePairedSFBScaleRemap = UsePairedBValueTmaRemap>
struct G1PairedMainloopRebind;

#if CUTLASS_VERSION >= 450
template <int Stages, int SchedulerPipelineStageCount, int AccumulatorPipelineStageCount,
          class ClusterShape, class ArchTag, class TileShape, class ElementA, class StridePairA,
          class ElementB, class StridePairB, class TiledMma, class GmemTiledCopyPairA,
          class SmemLayoutAtomA, class SmemCopyAtomA, class TransformA,
          class GmemTiledCopyPairB, class SmemLayoutAtomB, class SmemCopyAtomB,
          class TransformB, bool UsePairedBValueTmaRemap, bool UsePairedSFBScaleRemap>
struct G1PairedMainloopRebind<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm100ArrayTmaUmmaWarpSpecializedBlockwiseScaling<
        Stages, SchedulerPipelineStageCount, AccumulatorPipelineStageCount, ClusterShape,
        ArchTag>,
    TileShape, ElementA, StridePairA, ElementB, StridePairB, TiledMma, GmemTiledCopyPairA,
    SmemLayoutAtomA, SmemCopyAtomA, TransformA, GmemTiledCopyPairB, SmemLayoutAtomB,
    SmemCopyAtomB, TransformB>, UsePairedBValueTmaRemap, UsePairedSFBScaleRemap> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MxfpMainloopSm100ArrayTmaUmmaWarpSpecializedBlockwiseScaling<
          Stages, SchedulerPipelineStageCount, AccumulatorPipelineStageCount, ClusterShape,
          ArchTag, UsePairedBValueTmaRemap, UsePairedSFBScaleRemap>,
      TileShape, ElementA, StridePairA, ElementB, StridePairB, TiledMma, GmemTiledCopyPairA,
      SmemLayoutAtomA, SmemCopyAtomA, TransformA, GmemTiledCopyPairB, SmemLayoutAtomB,
      SmemCopyAtomB, TransformB>;
};
#else
template <int Stages, int SchedulerPipelineStageCount, int AccumulatorPipelineStageCount,
          class ClusterShape, class TileShape, class ElementA, class StridePairA,
          class ElementB, class StridePairB, class TiledMma, class GmemTiledCopyPairA,
          class SmemLayoutAtomA, class SmemCopyAtomA, class TransformA,
          class GmemTiledCopyPairB, class SmemLayoutAtomB, class SmemCopyAtomB,
          class TransformB, bool UsePairedBValueTmaRemap, bool UsePairedSFBScaleRemap>
struct G1PairedMainloopRebind<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm100ArrayTmaUmmaWarpSpecializedBlockwiseScaling<
        Stages, SchedulerPipelineStageCount, AccumulatorPipelineStageCount, ClusterShape>,
    TileShape, ElementA, StridePairA, ElementB, StridePairB, TiledMma, GmemTiledCopyPairA,
    SmemLayoutAtomA, SmemCopyAtomA, TransformA, GmemTiledCopyPairB, SmemLayoutAtomB,
    SmemCopyAtomB, TransformB>, UsePairedBValueTmaRemap, UsePairedSFBScaleRemap> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MxfpMainloopSm100ArrayTmaUmmaWarpSpecializedBlockwiseScaling<
          Stages, SchedulerPipelineStageCount, AccumulatorPipelineStageCount, ClusterShape,
          cutlass::arch::Sm100, UsePairedBValueTmaRemap, UsePairedSFBScaleRemap>,
      TileShape, ElementA, StridePairA, ElementB, StridePairB, TiledMma, GmemTiledCopyPairA,
      SmemLayoutAtomA, SmemCopyAtomA, TransformA, GmemTiledCopyPairB, SmemLayoutAtomB,
      SmemCopyAtomB, TransformB>;
};
#endif

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
          bool UseInputQuantShell = false, int MmaTileN = 128,
          bool UseG1PairedMainloopFork = false,
          bool UseG1PairedBValueTmaRemap = false,
          bool UseG1PairedSFBScaleRemap = UseG1PairedBValueTmaRemap,
          bool UseG1SwiGLUFusedNoGmemEpilogue = false>
cudaError_t launch_cutlass_blockscaled_grouped_ptr_gemm_sm100(
    void* arg_buffer, size_t arg_buffer_size_in_bytes, void* workspace,
    size_t workspace_size_in_bytes, DTypeA* a_ptr_base, DTypeB* b_ptr_base, float* sfa_ptr_base,
    float* sfb_ptr_base, DTypeOut* d_ptr_base, int* m_indptr, int max_m, int n, int k,
    int num_groups, cudaStream_t stream, const int* m_indptr_host = nullptr,
    const int* group_ids_host = nullptr, uint8_t* fused_c_fp8_base = nullptr,
    float* fused_c_scale_base = nullptr, const int* fused_padded_offsets_host = nullptr,
    int fused_intermediate = 0, int fused_pair_offset = 16, int fused_mode = 0) {
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
  using CollectiveEpilogue =
      std::conditional_t<UseG1SwiGLUFusedNoGmemEpilogue,
                         G1SwiGLUFusedNoGmemEpilogue<BaseCollectiveEpilogue>,
                         BaseCollectiveEpilogue>;

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
  using G1PairedForkMainloop =
      typename G1PairedMainloopRebind<BaseCollectiveMainloop,
                                      UseG1PairedBValueTmaRemap,
                                      UseG1PairedSFBScaleRemap>::type;
  using SelectedBaseCollectiveMainloop =
      std::conditional_t<UseG1PairedMainloopFork, G1PairedForkMainloop, BaseCollectiveMainloop>;
  using CollectiveMainloop =
      std::conditional_t<UseInputQuantShell, G2InputQuantCollective<SelectedBaseCollectiveMainloop>,
                         SelectedBaseCollectiveMainloop>;

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
  uint8_t** fused_c_ptr = nullptr;
  float** fused_c_scale_ptr = nullptr;
  if constexpr (UseG1SwiGLUFusedNoGmemEpilogue) {
    fused_c_ptr = allocate_aligned_buffer<uint8_t*>(int_cursor, int_remaining, num_groups);
    fused_c_scale_ptr = allocate_aligned_buffer<float*>(int_cursor, int_remaining, num_groups);
  }
  if (problem_sizes == nullptr || a_ptr == nullptr || b_ptr == nullptr || c_ptr == nullptr ||
      d_ptr == nullptr || sfa_ptr == nullptr || sfb_ptr == nullptr || stride_a == nullptr ||
      stride_b == nullptr || stride_c == nullptr || stride_d == nullptr ||
      layout_sfa == nullptr || layout_sfb == nullptr ||
      (UseG1SwiGLUFusedNoGmemEpilogue && (fused_c_ptr == nullptr || fused_c_scale_ptr == nullptr))) {
    return cudaErrorMemoryAllocation;
  }
  if constexpr (UseG1SwiGLUFusedNoGmemEpilogue) {
    if (m_indptr_host == nullptr || fused_padded_offsets_host == nullptr ||
        fused_c_fp8_base == nullptr || fused_c_scale_base == nullptr ||
        fused_intermediate <= 0) {
      return cudaErrorNotSupported;
    }
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
    std::array<uint8_t*, kMaxGroups> fused_c_ptr_host{};
    std::array<float*, kMaxGroups> fused_c_scale_ptr_host{};

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
      if constexpr (UseG1SwiGLUFusedNoGmemEpilogue) {
        int padded_offset = fused_padded_offsets_host[expert_id];
        fused_c_ptr_host[group] =
            fused_c_fp8_base + static_cast<int64_t>(padded_offset) * fused_intermediate;
        fused_c_scale_ptr_host[group] =
            fused_c_scale_base +
            static_cast<int64_t>(padded_offset) * (fused_intermediate / 128);
      }
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
    if constexpr (UseG1SwiGLUFusedNoGmemEpilogue) {
      cudaMemcpyAsync(fused_c_ptr, fused_c_ptr_host.data(), sizeof(fused_c_ptr_host[0]) * num_groups,
                      cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(fused_c_scale_ptr, fused_c_scale_ptr_host.data(),
                      sizeof(fused_c_scale_ptr_host[0]) * num_groups, cudaMemcpyHostToDevice,
                      stream);
    }
  } else {
    if constexpr (UseG1SwiGLUFusedNoGmemEpilogue) {
      return cudaErrorNotSupported;
    }
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
  if constexpr (UseInputQuantShell) {
    arguments.mainloop.base = {a_ptr, stride_a, b_ptr, stride_b, sfa_ptr, layout_sfa,
                               sfb_ptr, layout_sfb};
    arguments.mainloop.ptr_activation = nullptr;
  } else {
    arguments.mainloop = {a_ptr, stride_a, b_ptr, stride_b, sfa_ptr, layout_sfa,
                          sfb_ptr, layout_sfb};
  }
  if constexpr (UseG1SwiGLUFusedNoGmemEpilogue) {
    arguments.epilogue.base = {{}, c_ptr, stride_c, d_ptr, stride_d};
    arguments.epilogue.ptr_activation = fused_c_ptr;
    arguments.epilogue.ptr_activation_scale = fused_c_scale_ptr;
    arguments.epilogue.intermediate = fused_intermediate;
    arguments.epilogue.pair_offset = fused_pair_offset;
    arguments.epilogue.mode = fused_mode;
  } else {
    arguments.epilogue = {{}, c_ptr, stride_c, d_ptr, stride_d};
  }
  arguments.hw_info = hw_info;
  if constexpr (UseG1SwiGLUFusedNoGmemEpilogue) {
    arguments.epilogue.base.thread.alpha = 1.0f;
    arguments.epilogue.base.thread.beta = 0.0f;
  } else {
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;
  }

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
        ScaleGranularityM, ScaleMajorK, MmaTileM, MmaSM, DTypeA, DTypeB, DTypeOut, true, false,
        MmaTileN, false>(
        arg_buffer, arg_buffer_size_in_bytes, workspace, workspace_size_in_bytes, a_ptr_base,
        b_ptr_base, sfa_ptr_base, sfb_ptr_base, d_ptr_base, m_indptr, max_m, n, k, num_groups,
        stream, m_indptr_host, group_ids_host);
  }
  return launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
      ScaleGranularityM, ScaleMajorK, MmaTileM, MmaSM, DTypeA, DTypeB, DTypeOut, false, false,
      MmaTileN, false>(
      arg_buffer, arg_buffer_size_in_bytes, workspace, workspace_size_in_bytes, a_ptr_base,
      b_ptr_base, sfa_ptr_base, sfb_ptr_base, d_ptr_base, m_indptr, max_m, n, k, num_groups,
      stream, m_indptr_host, group_ids_host);
}

template <int ScaleGranularityM, bool ScaleMajorK, int MmaTileM, int MmaSM, typename DTypeA,
          typename DTypeB, typename DTypeOut, int MmaTileN = 128>
cudaError_t launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_g1_paired_fork(
    bool use_tma_epilogue, void* arg_buffer, size_t arg_buffer_size_in_bytes, void* workspace,
    size_t workspace_size_in_bytes, DTypeA* a_ptr_base, DTypeB* b_ptr_base,
    float* sfa_ptr_base, float* sfb_ptr_base, DTypeOut* d_ptr_base, int* m_indptr,
    int max_m, int n, int k, int num_groups, cudaStream_t stream,
    const int* m_indptr_host = nullptr, const int* group_ids_host = nullptr) {
  if (use_tma_epilogue) {
    return launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
        ScaleGranularityM, ScaleMajorK, MmaTileM, MmaSM, DTypeA, DTypeB, DTypeOut, true, false,
        MmaTileN, true>(
        arg_buffer, arg_buffer_size_in_bytes, workspace, workspace_size_in_bytes, a_ptr_base,
        b_ptr_base, sfa_ptr_base, sfb_ptr_base, d_ptr_base, m_indptr, max_m, n, k, num_groups,
        stream, m_indptr_host, group_ids_host);
  }
  return launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
      ScaleGranularityM, ScaleMajorK, MmaTileM, MmaSM, DTypeA, DTypeB, DTypeOut, false, false,
      MmaTileN, true>(
      arg_buffer, arg_buffer_size_in_bytes, workspace, workspace_size_in_bytes, a_ptr_base,
      b_ptr_base, sfa_ptr_base, sfb_ptr_base, d_ptr_base, m_indptr, max_m, n, k, num_groups,
      stream, m_indptr_host, group_ids_host);
}

template <int ScaleGranularityM, bool ScaleMajorK, int MmaTileM, int MmaSM, typename DTypeA,
          typename DTypeB, typename DTypeOut, int MmaTileN = 256>
cudaError_t launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_g1_paired_b_value_tma_remap(
    bool use_tma_epilogue, void* arg_buffer, size_t arg_buffer_size_in_bytes, void* workspace,
    size_t workspace_size_in_bytes, DTypeA* a_ptr_base, DTypeB* b_ptr_base,
    float* sfa_ptr_base, float* sfb_ptr_base, DTypeOut* d_ptr_base, int* m_indptr,
    int max_m, int n, int k, int num_groups, cudaStream_t stream,
    const int* m_indptr_host = nullptr, const int* group_ids_host = nullptr) {
  if (use_tma_epilogue) return cudaErrorNotSupported;
  return launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
      ScaleGranularityM, ScaleMajorK, MmaTileM, MmaSM, DTypeA, DTypeB, DTypeOut, false, false,
      MmaTileN, true, true>(
      arg_buffer, arg_buffer_size_in_bytes, workspace, workspace_size_in_bytes, a_ptr_base,
      b_ptr_base, sfa_ptr_base, sfb_ptr_base, d_ptr_base, m_indptr, max_m, n, k, num_groups,
      stream, m_indptr_host, group_ids_host);
}

template <int ScaleGranularityM, bool ScaleMajorK, int MmaTileM, int MmaSM, typename DTypeA,
          typename DTypeB, typename DTypeOut, int MmaTileN = 256>
cudaError_t
launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_g1_paired_b_value_tma_remap_interleaved_sfb(
    bool use_tma_epilogue, void* arg_buffer, size_t arg_buffer_size_in_bytes, void* workspace,
    size_t workspace_size_in_bytes, DTypeA* a_ptr_base, DTypeB* b_ptr_base,
    float* sfa_ptr_base, float* sfb_ptr_base, DTypeOut* d_ptr_base, int* m_indptr,
    int max_m, int n, int k, int num_groups, cudaStream_t stream,
    const int* m_indptr_host = nullptr, const int* group_ids_host = nullptr) {
  if (use_tma_epilogue) return cudaErrorNotSupported;
  return launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
      ScaleGranularityM, ScaleMajorK, MmaTileM, MmaSM, DTypeA, DTypeB, DTypeOut, false, false,
      MmaTileN, true, true, false>(
      arg_buffer, arg_buffer_size_in_bytes, workspace, workspace_size_in_bytes, a_ptr_base,
      b_ptr_base, sfa_ptr_base, sfb_ptr_base, d_ptr_base, m_indptr, max_m, n, k, num_groups,
      stream, m_indptr_host, group_ids_host);
}

template <int ScaleGranularityM, bool ScaleMajorK, int MmaTileM, int MmaSM, typename DTypeA,
          typename DTypeB, int MmaTileN = 256>
cudaError_t launch_cutlass_blockscaled_grouped_ptr_gemm_sm100_g1_swiglu_fused_poc(
    void* arg_buffer, size_t arg_buffer_size_in_bytes, void* workspace,
    size_t workspace_size_in_bytes, DTypeA* a_ptr_base, DTypeB* b_ptr_base,
    float* sfa_ptr_base, float* sfb_ptr_base, float* d_scratch_base, int* m_indptr,
    int max_m, int n, int k, int num_groups, cudaStream_t stream,
    const int* m_indptr_host, const int* group_ids_host, uint8_t* c_fp8_base,
    float* c_scale_base, const int* padded_offsets_host, int intermediate,
    int pair_offset = 16, int fused_mode = 0) {
  return launch_cutlass_blockscaled_grouped_ptr_gemm_sm100<
      ScaleGranularityM, ScaleMajorK, MmaTileM, MmaSM, DTypeA, DTypeB, float, false, false,
      MmaTileN, true, true, false, true>(
      arg_buffer, arg_buffer_size_in_bytes, workspace, workspace_size_in_bytes, a_ptr_base,
      b_ptr_base, sfa_ptr_base, sfb_ptr_base, d_scratch_base, m_indptr, max_m, n, k, num_groups,
      stream, m_indptr_host, group_ids_host, c_fp8_base, c_scale_base, padded_offsets_host,
      intermediate, pair_offset, fused_mode);
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
