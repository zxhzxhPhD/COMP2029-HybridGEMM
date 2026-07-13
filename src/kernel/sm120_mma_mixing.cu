#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/arch/mma_sm120.hpp"

namespace hygemm {

using W4A4Atom = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t,
    cutlass::float_e2m1_t,
    float,
    cutlass::float_ue8m0_t,
    32>;

using W4A8Atom = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e2m1_t,
    cutlass::float_e4m3_t,
    float,
    cutlass::float_ue8m0_t,
    32>;

static_assert(sizeof(W4A4Atom::CRegisters) == sizeof(W4A8Atom::CRegisters));
static_assert(sizeof(W4A4Atom::CRegisters) == 4 * sizeof(float));

enum class Mode : int {
  W4A4Only = 0,
  W4A8Only = 1,
  W4A4ThenW4A8 = 2,
  W4A8ThenW4A4 = 3,
};

constexpr int kThreads = 32;
constexpr int kAccumulatorRegisters = 4;
constexpr int kOutputCount = kThreads * kAccumulatorRegisters;

inline void check_cuda(cudaError_t error, char const* expression, char const* file, int line) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ':' << line << " for " << expression
              << ": " << cudaGetErrorString(error) << '\n';
    std::exit(1);
  }
}

#define HYGEMM_CUDA_CHECK(expression) \
  ::hygemm::check_cuda((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ void mma_w4a4(float (&accum)[4]) {
  // E2M1 +1.0 is nibble 0x2. The K64 instruction consumes packed FP4.
  constexpr uint32_t packed_e2m1_one = 0x22222222u;
  // Two K32 UE8M0 scales are packed in the K64 instruction's 16-bit SF register.
  // UE8M0 1.0 has exponent encoding 0x7f.
  constexpr uint16_t packed_scale_one = 0x7f7fu;

  W4A4Atom::fma(
      accum[0], accum[1], accum[2], accum[3],
      packed_e2m1_one, packed_e2m1_one, packed_e2m1_one, packed_e2m1_one,
      packed_e2m1_one, packed_e2m1_one,
      accum[0], accum[1], accum[2], accum[3],
      packed_scale_one, packed_scale_one);
}

__device__ __forceinline__ void mma_w4a8(float (&accum)[4]) {
  // The F8F6F4 K32 instruction holds FP4 in the middle four bits of each byte.
  // E2M1 +1.0 (0x2) therefore becomes byte 0x08 after the required <<2 shift.
  constexpr uint32_t unpacked_e2m1_one = 0x08080808u;
  // E4M3 +1.0 is byte 0x38.
  constexpr uint32_t packed_e4m3_one = 0x38383838u;
  constexpr uint8_t scale_one = 0x7fu;

  W4A8Atom::fma(
      accum[0], accum[1], accum[2], accum[3],
      unpacked_e2m1_one, unpacked_e2m1_one,
      unpacked_e2m1_one, unpacked_e2m1_one,
      packed_e4m3_one, packed_e4m3_one,
      accum[0], accum[1], accum[2], accum[3],
      scale_one, scale_one);
}

template <Mode mode>
__global__ void mma_mixing_kernel(float* output) {
  float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if constexpr (mode == Mode::W4A4Only) {
    mma_w4a4(accum);
  } else if constexpr (mode == Mode::W4A8Only) {
    mma_w4a8(accum);
  } else if constexpr (mode == Mode::W4A4ThenW4A8) {
    mma_w4a4(accum);
    mma_w4a8(accum);
  } else {
    mma_w4a8(accum);
    mma_w4a4(accum);
  }

  int lane = int(threadIdx.x);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    output[lane * 4 + i] = accum[i];
  }
}

struct CaseResult {
  std::string name;
  float expected;
  float minimum;
  float maximum;
  float max_abs_error;
  int mismatches;
};

template <Mode mode>
CaseResult run_case(char const* name, float expected, float* device_output) {
  mma_mixing_kernel<mode><<<1, kThreads>>>(device_output);
  HYGEMM_CUDA_CHECK(cudaGetLastError());
  HYGEMM_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> output(kOutputCount);
  HYGEMM_CUDA_CHECK(cudaMemcpy(
      output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CaseResult result{name, expected, output[0], output[0], 0.0f, 0};
  for (float value : output) {
    result.minimum = std::min(result.minimum, value);
    result.maximum = std::max(result.maximum, value);
    float error = std::abs(value - expected);
    result.max_abs_error = std::max(result.max_abs_error, error);
    result.mismatches += error == 0.0f ? 0 : 1;
  }
  return result;
}

void print_result(CaseResult const& result) {
  std::cout << "RESULT case=" << result.name
            << " expected=" << result.expected
            << " min=" << result.minimum
            << " max=" << result.maximum
            << " max_abs_error=" << result.max_abs_error
            << " mismatches=" << result.mismatches
            << " disposition=" << (result.mismatches == 0 ? "passed" : "failed")
            << '\n';
}

}  // namespace hygemm

int main() {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  std::cerr << "Binary was not compiled with SM120 MMA support.\n";
  return 1;
#else
  int device = 0;
  cudaDeviceProp props{};
  HYGEMM_CUDA_CHECK(cudaGetDevice(&device));
  HYGEMM_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  if (props.major != 12) {
    std::cerr << "SM120 GPU required; found compute capability "
              << props.major << '.' << props.minor << '\n';
    return 1;
  }

  float* device_output = nullptr;
  HYGEMM_CUDA_CHECK(cudaMalloc(&device_output, hygemm::kOutputCount * sizeof(float)));

  std::array<hygemm::CaseResult, 4> results{
      hygemm::run_case<hygemm::Mode::W4A4Only>("w4a4_only", 64.0f, device_output),
      hygemm::run_case<hygemm::Mode::W4A8Only>("w4a8_only", 32.0f, device_output),
      hygemm::run_case<hygemm::Mode::W4A4ThenW4A8>("w4a4_then_w4a8", 96.0f, device_output),
      hygemm::run_case<hygemm::Mode::W4A8ThenW4A4>("w4a8_then_w4a4", 96.0f, device_output),
  };

  HYGEMM_CUDA_CHECK(cudaFree(device_output));
  bool passed = true;
  for (auto const& result : results) {
    hygemm::print_result(result);
    passed &= result.mismatches == 0;
  }
  return passed ? 0 : 2;
#endif
}
