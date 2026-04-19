#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>

namespace mxfp {

struct MoeTcBackendConfig {
  int hidden;
  int intermediate;
  int block;
  int hidden_blocks;
  int intermediate_blocks;
  int gemm1_out_blocks;
};

class MoeTcBackend {
 public:
  virtual ~MoeTcBackend() = default;

  virtual bool IsAvailable() const = 0;
  virtual const char* BackendName() const = 0;

  virtual cudaError_t RunStep1Fused(const float* a_dev, int n_rows, const int* permuted_tok_e,
                                    const uint8_t* w13_e, const float* s13_e,
                                    float* c_perm_dev, cudaStream_t stream) = 0;

  virtual cudaError_t RunStep2(const float* c_perm_dev, int n_rows, const int* permuted_tok_e,
                               const float* permuted_w_e, const uint8_t* w2_e,
                               const float* s2_e, float* out_acc_dev,
                               cudaStream_t stream) = 0;
};

std::unique_ptr<MoeTcBackend> CreateMoeTcBackend5090Temp(const MoeTcBackendConfig& cfg);
std::unique_ptr<MoeTcBackend> CreateMoeTcBackendB200(const MoeTcBackendConfig& cfg);

}  // namespace mxfp
