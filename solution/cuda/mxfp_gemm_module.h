#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace mxfp {

// Device implementation for end-to-end GPU execution.
class DeviceMxfpGemmModule {
 public:
  DeviceMxfpGemmModule(int hidden, int intermediate, int block);
  ~DeviceMxfpGemmModule();

  DeviceMxfpGemmModule(const DeviceMxfpGemmModule&) = delete;
  DeviceMxfpGemmModule& operator=(const DeviceMxfpGemmModule&) = delete;

  void EnsureWorkspace(int64_t t, cudaStream_t stream);
  void RunExpert(const float* a_dev, int64_t t, const float* local_weight_dev, int local_expert_idx,
                 const uint8_t* gemm1_w_dev, const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
                 const float* gemm2_s_dev, float* out_acc_dev, cudaStream_t stream) const;

  // Compact / grouped path: runs the MoE expert over only the tokens that
  // were routed to it. Inputs `permuted_tok_e` and `permuted_w_e` are pointers
  // into the routing-metadata buffers, already offset to this expert's slice.
  // `n_rows` = number of tokens in that slice (= expert_counts[local_expert_idx]).
  // This is the swap point that a future CUTLASS SM100 blockwise FP8 grouped
  // GEMM would replace; the signature is already grouped-GEMM shaped.
  void RunExpertPermuted(const float* a_dev, int64_t t, int n_rows,
                         const int* permuted_tok_e, const float* permuted_w_e,
                         int local_expert_idx, const uint8_t* gemm1_w_dev,
                         const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
                         const float* gemm2_s_dev, float* out_acc_dev,
                         cudaStream_t stream) const;

  bool SupportsTcPath() const;

  // Blackwell/CUTLASS path: keep the current routing/per-expert loop, run
  // GEMM1 as direct FP8 block-scaled GEMM on compact routed rows, keep the
  // middle ops in FP32, quantize SwiGLU output back to block-scaled FP8, then
  // run GEMM2 as FP8 block-scaled CUTLASS.
  void RunExpertPermutedTc(const uint8_t* hidden_fp8_dev, const float* hidden_scale_dev,
                           int64_t t, int n_rows, const int* permuted_tok_e,
                           const float* permuted_w_e, int local_expert_idx,
                           const uint8_t* gemm1_w_dev, const float* gemm1_s_dev,
                           const uint8_t* gemm2_w_dev, const float* gemm2_s_dev,
                           float* out_acc_dev, cudaStream_t stream);
  void RunExpertPermutedTcToScratch(const uint8_t* hidden_fp8_dev,
                                    const float* hidden_scale_dev, int64_t t,
                                    int total_rows, int row_offset, int n_rows,
                                    const int* permuted_tok_e, int local_expert_idx,
                                    const uint8_t* gemm1_w_dev, const float* gemm1_s_dev,
                                    const uint8_t* gemm2_w_dev, const float* gemm2_s_dev,
                                    cudaStream_t stream, bool run_gemm2 = true);
  void ScatterTcScratch(int total_rows, const int* expert_offsets_dev,
                        const int* permuted_tok_dev, const int* permuted_expert_dev,
                        const float* permuted_w_dev, float* out_acc_dev, cudaStream_t stream);
  void WriteTcScratchToBf16Output(int64_t t, const int* routed_positions_dev,
                                  const int* routed_local_experts_dev,
                                  const float* routed_weights_dev, uint16_t* output_dev,
                                  cudaStream_t stream);
  bool RunGroupedGemm1ThenExpertGemm2Tc(
      const uint8_t* hidden_fp8_dev, const float* hidden_scale_dev, int64_t t, int total_rows,
      const int* expert_offsets_dev, const int* expert_counts_host,
      const int* expert_offsets_host, const int* permuted_tok_dev,
      const float* permuted_w_dev, const uint8_t* gemm1_w_dev,
      const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
      const float* gemm2_s_dev, float* out_acc_dev, cudaStream_t stream,
      bool direct_output = false, const int* routed_positions_dev = nullptr,
      const int* routed_local_experts_dev = nullptr, const float* routed_weights_dev = nullptr,
      uint16_t* output_dev = nullptr);
  bool RunExpertGemm1ThenGroupedGemm2Tc(
      const uint8_t* hidden_fp8_dev, const float* hidden_scale_dev, int64_t t, int total_rows,
      const int* expert_offsets_dev, const int* expert_counts_host,
      const int* expert_offsets_host, const int* permuted_tok_dev,
      const int* permuted_expert_dev, const float* permuted_w_dev, const uint8_t* gemm1_w_dev,
      const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
      const float* gemm2_s_dev, float* out_acc_dev, cudaStream_t stream);
  bool RunAllExpertsDenseTc(const uint8_t* hidden_fp8_dev, const float* hidden_scale_dev,
                            int64_t t, const int* routed_local_experts_dev,
                            const float* routed_weights_dev, const uint8_t* gemm1_w_dev,
                            const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
                            const float* gemm2_s_dev, uint16_t* output_dev,
                            cudaStream_t stream);

 private:
  void EnsureTcWorkspace(int rows);
  void EnsureTcActivationWorkspace(int rows);
  bool BuildInterleavedGemm1Runtime(const uint8_t* gemm1_w_dev, const float* gemm1_s_dev,
                                    cudaStream_t stream);
  void RunExpertGemm2TcFromG1(int64_t t, int total_rows, int n_rows, int g1_row_offset,
                              int scratch_row_offset, const int* permuted_tok_e,
                              const float* permuted_w_e,
                              int local_expert_idx, const uint8_t* gemm2_w_dev,
                              const float* gemm2_s_dev, float* out_acc_dev,
                              cudaStream_t stream, bool scatter_to_acc = true);
  void WriteTcPaddedScratchToBf16Output(int64_t t, const int* expert_offsets_dev,
                                        const int* padded_offsets_dev,
                                        const int* routed_positions_dev,
                                        const int* routed_local_experts_dev,
                                        const float* routed_weights_dev, uint16_t* output_dev,
                                        cudaStream_t stream);

  int hidden_;
  int intermediate_;
  int block_;
  int gemm1_out_;
  int hidden_blocks_;
  int intermediate_blocks_;
  int gemm1_out_blocks_;
  int64_t max_t_;
  bool emulate_fp8_unit_;
  bool emulate_fp16_operands_;
  bool emulate_acc_half_;
  bool tc_fp16_middle_;
  float* g1_dev_;
  float* c_dev_;
  int tc_max_rows_;
  bool tc_path_enabled_;
  uint8_t* tc_a_fp8_dev_;
  uint8_t* tc_b_col_dev_;
  float* tc_a_scale_dev_;
  float* tc_b_scale_dev_;
  float* tc_g1_f32_dev_;
  float* tc_act_f32_dev_;
  int tc_act_max_rows_;
  uint8_t* tc_c_fp8_dev_;
  float* tc_c_scale_dev_;
  float* tc_d_f32_dev_;
  int* tc_m_indptr_dev_;
  int* tc_m_indptr_host_;
  int* tc_padded_compact_rows_dev_;
  void* tc_int_workspace_dev_;
  void* tc_float_workspace_dev_;
  void* tc_group_int_workspace_dev_;
  void* tc_group_float_workspace_dev_;
  uint8_t* tc_g1_interleaved_w_dev_;
  float* tc_g1_interleaved_s_dev_;
};

}  // namespace mxfp
