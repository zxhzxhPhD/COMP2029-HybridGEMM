#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>

namespace hygemm::oc {

// One precision partition. Weight rows and their K32 scales have already been
// gathered into this partition's order. Both partitions share activation/scales.
// output[token * output_channels + channel_ids[row]] = dot(W[row], X[token]).
// channel_ids must be valid and injective across BOTH partitions. Empty
// partitions are omitted by the caller. Inputs must outlive all GPU launches.
struct DeviceProblem {
  int rows = 0, tokens = 0, input_channels = 0, output_channels = 0;
  void const* weight = nullptr;       // packed E2M1 or E4M3; row-major [rows,K]
  void const* activation = nullptr;   // E4M3; row-major [tokens,K]
  void const* weight_scales = nullptr;
  void const* activation_scales = nullptr; // CUTLASS Sm1xxBlockScaledConfig<32>
  int32_t const* channel_ids = nullptr;
  void* output = nullptr;             // FP16; row-major [tokens,output_channels]
};

struct KernelInfo {
  int id, tile_m, tile_n, tile_k, stages, registers_per_thread;
  int shared_bytes, local_bytes;
  bool fp8_weight, pingpong;
};

class Kernel {
 public:
  virtual ~Kernel() = default;
  virtual void run(cudaStream_t stream) = 0;
  virtual KernelInfo info() const = 0;
  // Rebind pointers for an unchanged problem shape without allocating workspace.
  // Serving adapters use this before graph capture/replay setup. Not all
  // implementations support rebinding; false means unsupported.
  virtual bool update_operands(DeviceProblem const&) { return false; }
};

constexpr int variant_count = 28;
// Throws std::runtime_error for an unsupported/invalid problem or CUDA failure.
std::unique_ptr<Kernel> make_kernel(int variant, bool fp8_weight,
                                  DeviceProblem const&, cudaStream_t stream);

} // namespace hygemm::oc
