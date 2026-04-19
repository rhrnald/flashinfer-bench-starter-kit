#include "moe_tc_backend.h"

#include <memory>

namespace mxfp {

namespace {

class MoeTcBackendB200Stub final : public MoeTcBackend {
 public:
  explicit MoeTcBackendB200Stub(const MoeTcBackendConfig& cfg) : cfg_(cfg) {}

  bool IsAvailable() const override {
    // TODO(B200): Wire runtime/library checks for SM100 tcgen05/TMA backend.
    return false;
  }

  const char* BackendName() const override { return "impl_b200_stub"; }

  cudaError_t RunStep1Fused(const float*, int, const int*, const uint8_t*, const float*, float*,
                            cudaStream_t) override {
    // TODO(B200): Implement TMA + tcgen05 fused GEMM1+SwiGLU.
    return cudaErrorNotSupported;
  }

  cudaError_t RunStep2(const float*, int, const int*, const float*, const uint8_t*, const float*,
                       float*, cudaStream_t) override {
    // TODO(B200): Implement blockwise FP8 GEMM2 + weighted scatter path.
    return cudaErrorNotSupported;
  }

 private:
  MoeTcBackendConfig cfg_;
};

}  // namespace

std::unique_ptr<MoeTcBackend> CreateMoeTcBackendB200Stub(const MoeTcBackendConfig& cfg) {
  return std::unique_ptr<MoeTcBackend>(new MoeTcBackendB200Stub(cfg));
}

}  // namespace mxfp
