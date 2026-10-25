#pragma once
#include "output_channel_gemm.h"

namespace hygemm::oc {
struct FusedInfo {
  KernelInfo kernel;
  uint32_t low_transaction_bytes, high_transaction_bytes;
  int virtual_rows;
};
class FusedKernel : public Kernel {
 public:
  virtual FusedInfo fused_info() const = 0;
  virtual bool update_problems(DeviceProblem const&,DeviceProblem const&) { return false; }
};
// Both problems describe the same X/Y and K/T/O; either weight partition may
// be empty. audit is optional device uint32_t[6], initially zero:
// low output tiles, high output tiles, producer switches, consumer switches,
// low K stages, high K stages. Disable it for timing. All pointer lifetimes
// and channel-map preconditions of DeviceProblem still apply.
std::unique_ptr<FusedKernel> make_fused_kernel(int variant, DeviceProblem const& low,
    DeviceProblem const& high, cudaStream_t stream, uint32_t* audit = nullptr);
} // namespace hygemm::oc
