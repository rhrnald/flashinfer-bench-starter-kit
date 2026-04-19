#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace mxfp {

class MoeTcBackend;

// Host-side FP8 e4m3fn decoder used by the current reference path.
float fp8_e4m3fn_to_float(uint8_t x);

// Host MXFP block-scale GEMM module.
// This isolates weight loading and GEMM math so we can later swap
// internals with B200 TC kernels while preserving call sites.
class HostMxfpGemmModule {
 public:
  HostMxfpGemmModule(int hidden, int intermediate, int block);

  size_t gemm1_weight_elems() const;
  size_t gemm1_scale_elems() const;
  size_t gemm2_weight_elems() const;
  size_t gemm2_scale_elems() const;

  void load_expert_from_device(int local_expert_idx, const uint8_t* gemm1_weights_dev,
                               const float* gemm1_scales_dev, const uint8_t* gemm2_weights_dev,
                               const float* gemm2_scales_dev, cudaStream_t stream);

  // GEMM1: [1, hidden] x [hidden, 2*intermediate] -> [2*intermediate]
  void gemm1_matvec(const float* a_row, float* g1_out) const;

  // SwiGLU: split g1 into [x1, x2], c = x1 * silu(x2)
  static void swiglu(const float* g1, int intermediate, float* c_out);

  // GEMM2 + accumulation: out_row += weight * ([1, intermediate] x [intermediate, hidden])
  void gemm2_matvec_accumulate(const float* c, float weight, float* out_row) const;

 private:
  int hidden_;
  int intermediate_;
  int block_;
  int gemm1_out_;
  int hidden_blocks_;
  int intermediate_blocks_;
  int gemm1_out_blocks_;

  // Host weights for current reference backend.
  // Layouts:
  // - w13_fp8_: [2I, H], w13_scale_: [2I/128, H/128]
  // - w2_fp8_:  [H, I],  w2_scale_:  [H/128, I/128]
  std::vector<uint8_t> w13_fp8_;
  std::vector<float> w13_scale_;
  std::vector<uint8_t> w2_fp8_;
  std::vector<float> w2_scale_;
};

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

  // Experimental Blackwell path: keep the current routing/per-expert loop, but
  // run each expert's two block-scale FP8 GEMMs through FlashInfer/CUTLASS
  // SM100 Tensor Core kernels. This consumes the original hidden FP8 tensor and
  // scale tensor directly, so GEMM1 avoids the scalar dequantized-FP32 path.
  void RunExpertPermutedTc(const uint8_t* hidden_fp8_dev, const float* hidden_scale_dev,
                           int64_t t, int n_rows, const int* permuted_tok_e,
                           const float* permuted_w_e, int local_expert_idx,
                           const uint8_t* gemm1_w_dev, const float* gemm1_s_dev,
                           const uint8_t* gemm2_w_dev, const float* gemm2_s_dev,
                           float* out_acc_dev, cudaStream_t stream);

 private:
  void EnsureTcWorkspace(int rows);

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
  bool tc5090_env_;
  std::unique_ptr<MoeTcBackend> tc5090_backend_;
  float* g1_dev_;
  float* c_dev_;
  int tc_max_rows_;
  bool tc_path_env_;
  uint8_t* tc_a_fp8_dev_;
  uint8_t* tc_b_col_dev_;
  float* tc_a_scale_dev_;
  float* tc_b_scale_dev_;
  uint16_t* tc_g1_bf16_dev_;
  uint8_t* tc_c_fp8_dev_;
  float* tc_c_scale_dev_;
  uint16_t* tc_d_bf16_dev_;
  int* tc_m_indptr_dev_;
  void* tc_int_workspace_dev_;
  void* tc_float_workspace_dev_;
};

}  // namespace mxfp
