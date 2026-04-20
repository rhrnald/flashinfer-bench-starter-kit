#pragma once

#include <cuda_runtime.h>

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
  void RunExpertPermuted(const uint8_t* hidden_fp8_dev, const float* hidden_scale_dev,
                         int64_t t, int n_rows,
                         const int* permuted_tok_e, const float* permuted_w_e,
                         int local_expert_idx, const uint8_t* gemm1_w_dev,
                         const float* gemm1_s_dev, const uint8_t* gemm2_w_dev,
                         const float* gemm2_s_dev, float* out_acc_dev,
                         cudaStream_t stream) const;

  // Step1 direct launch over all local experts at once.
  // expert_t_valid: length-32 counts per local expert
  // expert_offset: length-32 start offsets into valid_token_idx
  // valid_token_idx: concatenated routed token ids
  cudaError_t RunStep1AllExpertsDirect(const uint8_t* hidden_fp8_dev,
                                       const float* hidden_scale_dev, int64_t t,
                                       const int* expert_t_valid,
                                       const int* expert_offset,
                                       const int* valid_token_idx,
                                       const uint8_t* gemm1_w_dev,
                                       const float* gemm1_s_dev,
                                       float* c_perm_all_dev,
                                       cudaStream_t stream) const;

  // Step2-only runner using precomputed compact Step1 output [n_rows, I].
  void RunStep2PermutedOnly(const float* c_perm_e, int n_rows,
                            const int* permuted_tok_e, const float* permuted_w_e,
                            int local_expert_idx, const uint8_t* gemm2_w_dev,
                            const float* gemm2_s_dev, float* out_acc_dev,
                            cudaStream_t stream) const;

  cudaError_t RunStep2AllExpertsDirect(const float* c_perm_all_dev,
                                       const int* expert_t_valid,
                                       const int* expert_offset,
                                       const int* valid_token_idx,
                                       const float* valid_token_w,
                                       const uint8_t* gemm2_w_dev,
                                       const float* gemm2_s_dev,
                                       float* out_acc_dev,
                                       cudaStream_t stream) const;

  bool IsB200DirectEnabled() const;

 private:
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
  bool quantize_scale_e8m0_;
  bool b200_direct_enabled_;
  mutable void* step1_hidden_tma_desc_dev_;
  mutable void* step1_w13_tma_desc_dev_;
  mutable void* step2_w2_tma_desc_dev_;
  float* g1_dev_;
  float* c_dev_;
};

}  // namespace mxfp
