#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mxfp {

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
  float* g1_dev_;
  float* c_dev_;
};

}  // namespace mxfp
