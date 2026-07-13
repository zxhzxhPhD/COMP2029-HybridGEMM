#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/arch/mma_sm120.hpp"

namespace hygemm {

using W4A4K64Atom = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue8m0_t, 32>;
using W4A4K32Atom = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue8m0_t, 32>;
using W4A8K32Atom = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e4m3_t, float,
    cutlass::float_ue8m0_t, 32>;

enum class Policy : int {
  LLFast,
  LLGeneric,
  HH,
  LH,
  HL,
  SingletonGrouped,
};

inline void check_cuda(cudaError_t error, char const* expression, char const* file, int line) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ':' << line << " for " << expression
              << ": " << cudaGetErrorString(error) << '\n';
    std::exit(1);
  }
}

#define HYGEMM_CUDA_CHECK(expression) \
  ::hygemm::check_cuda((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ void mma_low_k64(float (&accum)[4]) {
  constexpr uint32_t fp4_one = 0x22222222u;
  constexpr uint16_t sf_pair_one = 0x7f7fu;
  W4A4K64Atom::fma(
      accum[0], accum[1], accum[2], accum[3],
      fp4_one, fp4_one, fp4_one, fp4_one, fp4_one, fp4_one,
      accum[0], accum[1], accum[2], accum[3],
      sf_pair_one, sf_pair_one);
}

__device__ __forceinline__ void mma_low_k32(float (&accum)[4]) {
  constexpr uint32_t fp4_one_shifted = 0x08080808u;
  constexpr uint8_t sf_one = 0x7fu;
  W4A4K32Atom::fma(
      accum[0], accum[1], accum[2], accum[3],
      fp4_one_shifted, fp4_one_shifted, fp4_one_shifted, fp4_one_shifted,
      fp4_one_shifted, fp4_one_shifted,
      accum[0], accum[1], accum[2], accum[3], sf_one, sf_one);
}

__device__ __forceinline__ void mma_high_k32(float (&accum)[4]) {
  constexpr uint32_t fp4_one_shifted = 0x08080808u;
  constexpr uint32_t fp8_one = 0x38383838u;
  constexpr uint8_t sf_one = 0x7fu;
  W4A8K32Atom::fma(
      accum[0], accum[1], accum[2], accum[3],
      fp4_one_shifted, fp4_one_shifted, fp4_one_shifted, fp4_one_shifted,
      fp8_one, fp8_one,
      accum[0], accum[1], accum[2], accum[3], sf_one, sf_one);
}

template <Policy policy>
__device__ __forceinline__ void execute_pair(float (&accum)[4]) {
  if constexpr (policy == Policy::LLFast) {
    mma_low_k64(accum);
  } else if constexpr (policy == Policy::LLGeneric) {
    mma_low_k32(accum);
    mma_low_k32(accum);
  } else if constexpr (policy == Policy::HH) {
    mma_high_k32(accum);
    mma_high_k32(accum);
  } else if constexpr (policy == Policy::LH) {
    mma_low_k32(accum);
    mma_high_k32(accum);
  } else {
    mma_high_k32(accum);
    mma_low_k32(accum);
  }
}

template <Policy policy, int Chains>
__global__ __launch_bounds__(256) void policy_kernel(float* checksums, int iterations) {
  float accum[Chains][4]{};

  if constexpr (policy == Policy::SingletonGrouped) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
      for (int chain = 0; chain < Chains; ++chain) {
        mma_low_k32(accum[chain]);
      }
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
      for (int chain = 0; chain < Chains; ++chain) {
        mma_high_k32(accum[chain]);
      }
    }
  } else {
    for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
      for (int chain = 0; chain < Chains; ++chain) {
        execute_pair<policy>(accum[chain]);
      }
    }
  }

  float checksum = 0.0f;
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
      checksum += accum[chain][reg];
    }
  }
  checksums[blockIdx.x * blockDim.x + threadIdx.x] = checksum;
}

struct Result {
  std::string policy;
  int chains;
  float runtime_ms;
  double gpairs_per_second;
  int registers_per_thread;
  int mismatches;
  float max_abs_error;
};

template <Policy policy, int Chains>
Result benchmark(
    char const* name, float* device_checksums, int output_count,
    int grid, int block, int iterations, int launches) {
  cudaFuncAttributes attributes{};
  HYGEMM_CUDA_CHECK(cudaFuncGetAttributes(&attributes, policy_kernel<policy, Chains>));

  policy_kernel<policy, Chains><<<grid, block>>>(device_checksums, iterations);
  HYGEMM_CUDA_CHECK(cudaGetLastError());
  HYGEMM_CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  HYGEMM_CUDA_CHECK(cudaEventCreate(&start));
  HYGEMM_CUDA_CHECK(cudaEventCreate(&stop));
  HYGEMM_CUDA_CHECK(cudaEventRecord(start));
  for (int launch = 0; launch < launches; ++launch) {
    policy_kernel<policy, Chains><<<grid, block>>>(device_checksums, iterations);
  }
  HYGEMM_CUDA_CHECK(cudaEventRecord(stop));
  HYGEMM_CUDA_CHECK(cudaEventSynchronize(stop));
  float total_ms = 0.0f;
  HYGEMM_CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  HYGEMM_CUDA_CHECK(cudaEventDestroy(start));
  HYGEMM_CUDA_CHECK(cudaEventDestroy(stop));

  std::vector<float> checksums(output_count);
  HYGEMM_CUDA_CHECK(cudaMemcpy(
      checksums.data(), device_checksums, output_count * sizeof(float), cudaMemcpyDeviceToHost));
  float expected = float(Chains * 4 * 64 * iterations);
  int mismatches = 0;
  float max_abs_error = 0.0f;
  for (float value : checksums) {
    float error = std::abs(value - expected);
    max_abs_error = std::max(max_abs_error, error);
    mismatches += error == 0.0f ? 0 : 1;
  }

  double warp_pairs = double(grid) * (block / 32) * iterations * Chains * launches;
  double gpairs_per_second = warp_pairs / (double(total_ms) * 1.0e6);
  return {name, Chains, total_ms / launches, gpairs_per_second,
          attributes.numRegs, mismatches, max_abs_error};
}

void print_result(Result const& result, double ll_fast_rate) {
  std::cout << std::fixed << std::setprecision(4)
            << "RESULT policy=" << result.policy
            << " chains=" << result.chains
            << " runtime_ms=" << result.runtime_ms
            << " gpairs_s=" << result.gpairs_per_second
            << " relative_to_ll_fast=" << result.gpairs_per_second / ll_fast_rate
            << " registers=" << result.registers_per_thread
            << " mismatches=" << result.mismatches
            << " max_abs_error=" << result.max_abs_error
            << " disposition=" << (result.mismatches == 0 ? "passed" : "failed")
            << '\n';
}

template <int Chains>
bool run_suite(float* device_checksums, int output_count, int grid, int block,
               int iterations, int launches) {
  Result ll_fast = benchmark<Policy::LLFast, Chains>(
      "ll_k64_omma", device_checksums, output_count, grid, block, iterations, launches);
  Result ll_generic = benchmark<Policy::LLGeneric, Chains>(
      "ll_2xk32_qmma", device_checksums, output_count, grid, block, iterations, launches);
  Result hh = benchmark<Policy::HH, Chains>(
      "hh_2xk32_qmma", device_checksums, output_count, grid, block, iterations, launches);
  Result lh = benchmark<Policy::LH, Chains>(
      "lh_k32_fallback", device_checksums, output_count, grid, block, iterations, launches);
  Result hl = benchmark<Policy::HL, Chains>(
      "hl_k32_fallback", device_checksums, output_count, grid, block, iterations, launches);
  Result singleton_grouped = benchmark<Policy::SingletonGrouped, Chains>(
      "singleton_grouped_streams", device_checksums, output_count,
      grid, block, iterations, launches);

  print_result(ll_fast, ll_fast.gpairs_per_second);
  print_result(ll_generic, ll_fast.gpairs_per_second);
  print_result(hh, ll_fast.gpairs_per_second);
  print_result(lh, ll_fast.gpairs_per_second);
  print_result(hl, ll_fast.gpairs_per_second);
  print_result(singleton_grouped, ll_fast.gpairs_per_second);
  return ll_fast.mismatches == 0 && ll_generic.mismatches == 0 && hh.mismatches == 0 &&
         lh.mismatches == 0 && hl.mismatches == 0 && singleton_grouped.mismatches == 0;
}

}  // namespace hygemm

int main(int argc, char** argv) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  std::cerr << "Binary was not compiled with SM120 MMA support.\n";
  return 1;
#else
  int iterations = argc > 1 ? std::atoi(argv[1]) : 512;
  int launches = argc > 2 ? std::atoi(argv[2]) : 20;
  if (iterations <= 0 || launches <= 0) {
    std::cerr << "Usage: " << argv[0] << " [iterations] [launches]\n";
    return 1;
  }

  int device = 0;
  cudaDeviceProp props{};
  HYGEMM_CUDA_CHECK(cudaGetDevice(&device));
  HYGEMM_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  if (props.major != 12) {
    std::cerr << "SM120 GPU required.\n";
    return 1;
  }

  constexpr int block = 256;
  int grid = props.multiProcessorCount * 8;
  int output_count = grid * block;
  float* device_checksums = nullptr;
  HYGEMM_CUDA_CHECK(cudaMalloc(&device_checksums, output_count * sizeof(float)));

  std::cout << "CONFIG gpu=" << props.name
            << " sms=" << props.multiProcessorCount
            << " grid=" << grid << " block=" << block
            << " iterations=" << iterations << " launches=" << launches << '\n';
  bool passed = true;
  passed &= hygemm::run_suite<1>(
      device_checksums, output_count, grid, block, iterations, launches);
  passed &= hygemm::run_suite<4>(
      device_checksums, output_count, grid, block, iterations, launches);

  HYGEMM_CUDA_CHECK(cudaFree(device_checksums));
  return passed ? 0 : 2;
#endif
}
