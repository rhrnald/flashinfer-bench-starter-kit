#pragma once

// Drop-in replacement for the current mma_64x8x64_tcgen05() implementation.
//
// Assumptions:
// - The surrounding file already defines:
//     * LaneAccum8x1
//     * stage_tcgen_operands(...)
//     * mma_64x8x64_tt_fallback(...)
//     * kStep1* compile-time constants used below
// - The surrounding file already includes the same CuTe/CUTLASS headers as the
//   user's latest kernel.
//
// Design choice:
// - tcgen05.mma itself is issued through inline PTX.
// - TMEM allocation, mbarrier wait, and TMEM->register copy keep using the
//   already-working CUTLASS/CuTe helpers, because those pieces are mostly about
//   address mapping rather than MMA issue.
//
// This keeps the risky part small:
//   SMEM A/B -> inline PTX tcgen05.mma -> TMEM -> CuTe tcgen05.ld copy -> RMEM
//   -> multiply by gate/up/hidden scales -> accumulate into LaneAccum8x1.

namespace direct_backend {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

__device__ __forceinline__ uint64_t pack_umma_desc_u64(cute::UMMA::SmemDescriptor const& desc) {
  return (static_cast<uint64_t>(desc.hi) << 32) | static_cast<uint64_t>(desc.lo);
}

__device__ __forceinline__ constexpr uint32_t make_tcgen05_idesc_f8f6f4_f32_dense(int m, int n) {
  // PTX ISA Table 42 (.kind::f8f6f4):
  // bits  4:5   dtype  = F32 -> 1
  // bits  7:9   atype  = E4M3 -> 0
  // bits 10:12  btype  = E4M3 -> 0
  // bit  15     transpose A = 0
  // bit  16     transpose B = 0
  // bits 17:22  N >> 3
  // bits 24:28  M >> 4
  // everything else 0 for dense non-negated non-sparse MMA.
  return (1u << 4) | (static_cast<uint32_t>(n >> 3) << 17) | (static_cast<uint32_t>(m >> 4) << 24);
}

__device__ __forceinline__ void tcgen05_mma_f8f6f4_cta1_ss(
    uint32_t d_tmem,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t idesc,
    bool enable_input_d) {
  uint32_t enable_u32 = enable_input_d ? 1u : 0u;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.u32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
      "}\n"
      :
      : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_u32)
      : "memory");
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
    uint32_t tmem_base_ptr,
    uint64_t* mma_barrier,
    int& mma_phase_bit) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using Traits = cute::MMA_Traits<
      cute::SM100_MMA_F8F6F4_SS,
      uint8_t,
      uint8_t,
      float,
      cute::C<kStep1TcgenM>,
      cute::C<kStep1TcgenN>,
      cute::integral_constant<cute::UMMA::Major, cute::UMMA::Major::K>,
      cute::integral_constant<cute::UMMA::Major, cute::UMMA::Major::K>,
      cute::integral_constant<cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>,
      cute::integral_constant<cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>>;
  using TiledMma = decltype(cute::make_tiled_mma(Traits{}));

  TiledMma tiled_mma = cute::make_tiled_mma(Traits{});

  // Keep CuTe only for the canonical SMEM layout and TMEM->RMEM mapping.
  auto mma_shape_A = cute::partition_shape_A(
      tiled_mma, cute::make_shape(cute::Int<kStep1TcgenM>{}, cute::Int<kStep1TcgenSubKBytes>{}));
  auto mma_shape_B = cute::partition_shape_B(
      tiled_mma, cute::make_shape(cute::Int<kStep1TcgenN>{}, cute::Int<kStep1TcgenSubKBytes>{}));
  auto sA_layout = cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW128_Atom<uint8_t>{}, mma_shape_A);
  auto sB_layout = cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW128_Atom<uint8_t>{}, mma_shape_B);

  auto sA_tcgen = cute::make_tensor(cute::make_smem_ptr(smem_A_tcgen_bytes), sA_layout);
  auto sB_tcgen = cute::make_tensor(cute::make_smem_ptr(smem_B_tcgen_bytes), sB_layout);

  stage_tcgen_operands(
      sA_tcgen,
      sB_tcgen,
      reinterpret_cast<const uint8_t*>(smem_A_group_combined_words),
      reinterpret_cast<const uint8_t*>(smem_B_group_words),
      k_blk_local,
      k64_sub,
      lane);
  __syncthreads();

  auto descA = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(sA_tcgen);
  auto descB = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(sB_tcgen);
  const uint64_t a_desc = pack_umma_desc_u64(descA);
  const uint64_t b_desc = pack_umma_desc_u64(descB);
  constexpr uint32_t idesc = make_tcgen05_idesc_f8f6f4_f32_dense(kStep1TcgenM, kStep1TcgenN);

  auto c_layout = cute::make_layout(
      cute::make_shape(cute::Int<kStep1TcgenM>{}, cute::Int<kStep1TcgenN>{}),
      cute::make_stride(cute::Int<kStep1TcgenN>{}, cute::Int<1>{}));
  auto mC = cute::make_tensor(cute::make_gmem_ptr(static_cast<float*>(nullptr)), c_layout);
  auto cId = cute::make_identity_tensor(cute::make_shape(cute::Int<kStep1TcgenM>{}, cute::Int<kStep1TcgenN>{}));
  auto cta_mma = tiled_mma.get_slice(0);
  auto tCgC = cta_mma.partition_C(mC);
  auto tCcC = cta_mma.partition_C(cId);
  auto tCtAcc = cta_mma.make_fragment_C(tCgC);
  tCtAcc.data() = tmem_base_ptr;

  // One thread is enough to issue tcgen05.mma for cta_group::1.
  // We intentionally use enable_input_d = false because the final accumulation
  // happens in registers after scaling, not in TMEM across calls.
  if (lane == 0) {
    tcgen05_mma_f8f6f4_cta1_ss(tmem_base_ptr, a_desc, b_desc, idesc, /*enable_input_d=*/false);
    cutlass::arch::umma_arrive(mma_barrier);
  }

  // Reuse the existing barrier/wait path that was already working in the CuTe version.
  cute::wait_barrier(*mma_barrier, mma_phase_bit);
  mma_phase_bit ^= 1;

  // Reuse the already-correct TMEM->RMEM copy mapping.
  auto tiled_t2r_copy = cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);
  auto tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  auto tDgC = thr_t2r_copy.partition_D(tCgC);
  auto tDcC = thr_t2r_copy.partition_D(tCcC);
  using AccType = typename decltype(tCtAcc)::value_type;
  auto tDrAcc = cute::make_tensor<AccType>(cute::shape(tDgC));
  cute::copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  float gate_fused_scale[kStep1RowTile];
  float up_fused_scale[kStep1RowTile];
#pragma unroll
  for (int rr = 0; rr < kStep1RowTile; ++rr) {
    if (rr < n_rows) {
      const int packed_idx = packed_base + row_tile + rr;
      const int token_idx = valid_token_idx[packed_idx];
      const float hidden_block_scale =
          hidden_scale_dev[static_cast<int64_t>(k_blk) * t + token_idx];
      gate_fused_scale[rr] = gate_block_scale * hidden_block_scale;
      up_fused_scale[rr] = up_block_scale * hidden_block_scale;
    } else {
      gate_fused_scale[rr] = 0.0f;
      up_fused_scale[rr] = 0.0f;
    }
  }

#pragma unroll
  for (int i = 0; i < cute::size(tDrAcc); ++i) {
    auto coord = tDcC(i);
    const int m = static_cast<int>(cute::get<0>(coord));
    const int n = static_cast<int>(cute::get<1>(coord));
    if (n >= n_rows) continue;
    const float val = tDrAcc(i);
    if (m < kStep1OutRowsPerCta) {
      if (m == lane) {
        gate_acc.v[n] += val * gate_fused_scale[n];
      }
    } else {
      if (m - kStep1OutRowsPerCta == lane) {
        up_acc.v[n] += val * up_fused_scale[n];
      }
    }
  }
  __syncthreads();
#else
  mma_64x8x64_tt_fallback(
      gate_acc,
      up_acc,
      smem_A_group_combined_words,
      smem_B_group_words,
      gate_block_scale,
      up_block_scale,
      hidden_scale_dev,
      t,
      valid_token_idx,
      packed_base,
      row_tile,
      n_rows,
      k_blk,
      k_blk_local,
      k64_sub,
      lane);
#endif
}

}  // namespace direct_backend
