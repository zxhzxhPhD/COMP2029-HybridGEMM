#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "sm120_k64_paired_hybrid_collective.hpp"

using namespace cute;

namespace hygemm {

inline void check_cuda(cudaError_t error, char const* expression, char const* file, int line) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ':' << line << " for " << expression
              << ": " << cudaGetErrorString(error) << '\n';
    std::exit(1);
  }
}

inline void check_cutlass(
    cutlass::Status status, char const* expression, char const* file, int line) {
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS error at " << file << ':' << line << " for " << expression
              << ": " << cutlassGetStatusString(status) << '\n';
    std::exit(1);
  }
}

#define HYGEMM_CUDA_CHECK(expression) \
  ::hygemm::check_cuda((expression), #expression, __FILE__, __LINE__)
#define HYGEMM_CUTLASS_CHECK(expression) \
  ::hygemm::check_cutlass((expression), #expression, __FILE__, __LINE__)

struct Options {
  int m = 256;
  int n = 128;
  int k = 512;
  int low_k = 256;
  int warmup = 3;
  int iterations = 20;
  std::string mode = "both";

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("low-k", low_k);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("mode", mode);
  }

  bool valid() const {
    return m > 0 && n > 0 && k > 0 && low_k > 0 && low_k < k &&
           k % 128 == 0 && low_k % 128 == 0 &&
           warmup >= 0 && iterations > 0 &&
           (mode == "both" || mode == "2a" || mode == "2b");
  }
};

template <bool CompactStreams>
struct HybridConfig {
  using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  using ElementBLow = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  using ElementBHigh = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementC = float;
  using ElementD = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<typename ElementA::DataType>::value;
  static constexpr int AlignmentBLow = 128 / cutlass::sizeof_bits<typename ElementBLow::DataType>::value;
  static constexpr int AlignmentBHigh = 128 / cutlass::sizeof_bits<typename ElementBHigh::DataType>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using TileShape = Shape<_128, _64, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
          TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute,
          ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  template <class ElementB, int AlignmentB>
  using StockCollective =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
          ElementA, LayoutA, AlignmentA,
          ElementB, LayoutB, AlignmentB,
          ElementAccumulator,
          TileShape, ClusterShape,
          cutlass::gemm::collective::StageCount<2>,
          cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using LowCollective = StockCollective<ElementBLow, AlignmentBLow>;
  using HighCollective = StockCollective<ElementBHigh, AlignmentBHigh>;
  using CollectiveMainloop = Sm120K64PairedHybridCollective<
      LowCollective, HighCollective, CompactStreams>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using LowGemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, LowCollective, CollectiveEpilogue>;
  using HighGemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, HighCollective, CollectiveEpilogue>;
  using LowGemm = cutlass::gemm::device::GemmUniversalAdapter<LowGemmKernel>;
  using HighGemm = cutlass::gemm::device::GemmUniversalAdapter<HighGemmKernel>;
};

template <class T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

template <class Tensor>
void fill(Tensor& tensor, typename Tensor::Element value) {
  cutlass::reference::host::TensorFill(tensor.host_view(), value);
  tensor.sync_device();
}

template <class Tensor, class Layout, class Element>
void fill_k64_pair(Tensor& tensor, Layout const& layout, int outer_extent,
                   int destination_pair, Element value) {
  int destination_k = destination_pair * 64;
  for (int outer = 0; outer < outer_extent; ++outer) {
    for (int k = 0; k < 64; ++k) {
      auto offset = int(layout(outer, destination_k + k, 0));
      tensor.host_data(offset) = value;
    }
  }
}

template <bool CompactStreams>
bool run_hybrid(Options const& options) {
  using Config = HybridConfig<CompactStreams>;
  using Gemm = typename Config::Gemm;
  using LowCollective = typename Config::LowCollective;
  using HighCollective = typename Config::HighCollective;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutA = decltype(make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutB = decltype(make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutC = decltype(make_layout(make_shape(0, 0, 0), StrideC{}));
  using LayoutD = decltype(make_layout(make_shape(0, 0, 0), StrideD{}));

  int high_k = options.k - options.low_k;
  int low_physical_k = CompactStreams ? options.low_k : options.k;
  int high_physical_k = CompactStreams ? high_k : options.k;

  StrideA stride_a_low = cutlass::make_cute_packed_stride(
      StrideA{}, {options.m, low_physical_k, 1});
  StrideA stride_a_high = cutlass::make_cute_packed_stride(
      StrideA{}, {options.m, high_physical_k, 1});
  StrideB stride_b_low = cutlass::make_cute_packed_stride(
      StrideB{}, {options.n, low_physical_k, 1});
  StrideB stride_b_high = cutlass::make_cute_packed_stride(
      StrideB{}, {options.n, high_physical_k, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(
      StrideC{}, {options.m, options.n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, {options.m, options.n, 1});

  LayoutA layout_a_low = make_layout(
      make_shape(options.m, low_physical_k, 1), stride_a_low);
  LayoutA layout_a_high = make_layout(
      make_shape(options.m, high_physical_k, 1), stride_a_high);
  LayoutB layout_b_low = make_layout(
      make_shape(options.n, low_physical_k, 1), stride_b_low);
  LayoutB layout_b_high = make_layout(
      make_shape(options.n, high_physical_k, 1), stride_b_high);
  LayoutC layout_c = make_layout(make_shape(options.m, options.n, 1), stride_c);
  LayoutD layout_d = make_layout(make_shape(options.m, options.n, 1), stride_d);

  auto low_problem = make_shape(options.m, options.n, low_physical_k, 1);
  auto high_problem = make_shape(options.m, options.n, high_physical_k, 1);
  auto layout_sfa_low = LowCollective::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(low_problem);
  auto layout_sfb_low = LowCollective::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(low_problem);
  auto layout_sfa_high = HighCollective::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(high_problem);
  auto layout_sfb_high = HighCollective::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(high_problem);

  cutlass::HostTensor<cutlass::float_e2m1_t, cutlass::layout::PackedVectorLayout>
      block_a_low(cutlass::make_Coord(size(layout_a_low)));
  cutlass::HostTensor<cutlass::float_e2m1_t, cutlass::layout::PackedVectorLayout>
      block_a_high(cutlass::make_Coord(size(layout_a_high)));
  cutlass::HostTensor<cutlass::float_e2m1_t, cutlass::layout::PackedVectorLayout>
      block_b_low(cutlass::make_Coord(size(layout_b_low)));
  cutlass::HostTensor<cutlass::float_e4m3_t, cutlass::layout::PackedVectorLayout>
      block_b_high(cutlass::make_Coord(size(layout_b_high)));
  cutlass::HostTensor<cutlass::float_ue8m0_t, cutlass::layout::PackedVectorLayout>
      block_sfa_low(cutlass::make_Coord(size(filter_zeros(layout_sfa_low))));
  cutlass::HostTensor<cutlass::float_ue8m0_t, cutlass::layout::PackedVectorLayout>
      block_sfb_low(cutlass::make_Coord(size(filter_zeros(layout_sfb_low))));
  cutlass::HostTensor<cutlass::float_ue8m0_t, cutlass::layout::PackedVectorLayout>
      block_sfa_high(cutlass::make_Coord(size(filter_zeros(layout_sfa_high))));
  cutlass::HostTensor<cutlass::float_ue8m0_t, cutlass::layout::PackedVectorLayout>
      block_sfb_high(cutlass::make_Coord(size(filter_zeros(layout_sfb_high))));
  cutlass::HostTensor<float, cutlass::layout::PackedVectorLayout>
      block_c(cutlass::make_Coord(size(layout_c)));
  cutlass::HostTensor<float, cutlass::layout::PackedVectorLayout>
      block_d(cutlass::make_Coord(size(layout_d)));

  fill(block_a_low, cutlass::float_e2m1_t(1.0f));
  fill(block_a_high, cutlass::float_e2m1_t(1.0f));
  fill(block_b_low, cutlass::float_e2m1_t(1.0f));
  fill(block_b_high, cutlass::float_e4m3_t(2.0f));
  fill(block_sfa_low, cutlass::float_ue8m0_t(1.0f));
  fill(block_sfb_low, cutlass::float_ue8m0_t(1.0f));
  fill(block_sfa_high, cutlass::float_ue8m0_t(1.0f));
  fill(block_sfb_high, cutlass::float_ue8m0_t(1.0f));
  fill(block_c, 0.0f);

  float expected_low = float(low_physical_k);
  float expected_high = float(2 * high_physical_k);
  float expected = float(options.low_k + 2 * high_k);
  std::vector<int> high_pair_ids;
  if constexpr (CompactStreams) {
    // Reference pack for a non-contiguous, layer-static K64 mask. The compact
    // streams preserve the selected logical K64 atoms but remove all gaps, so
    // the mainloop itself only performs continuous TMA traversal.
    int pair_count = options.k / 64;
    int high_pair_count = high_k / 64;
    std::vector<bool> high_mask(pair_count, false);
    for (int parity = 1; parity >= 0 && int(high_pair_ids.size()) < high_pair_count; --parity) {
      for (int pair = parity; pair < pair_count && int(high_pair_ids.size()) < high_pair_count;
           pair += 2) {
        high_mask[pair] = true;
        high_pair_ids.push_back(pair);
      }
    }

    int low_destination = 0;
    int high_destination = 0;
    expected_low = 0.0f;
    expected_high = 0.0f;
    for (int pair = 0; pair < pair_count; ++pair) {
      float weight = 0.5f * float(1 + (pair % 4));
      float activation = 0.5f * float(1 + ((pair * 3) % 4));
      float pair_contribution = 64.0f * weight * activation;
      if (high_mask[pair]) {
        expected_high += pair_contribution;
        fill_k64_pair(block_a_high, layout_a_high, options.m, high_destination,
                      cutlass::float_e2m1_t(weight));
        fill_k64_pair(block_b_high, layout_b_high, options.n, high_destination,
                      cutlass::float_e4m3_t(activation));
        ++high_destination;
      } else {
        expected_low += pair_contribution;
        fill_k64_pair(block_a_low, layout_a_low, options.m, low_destination,
                      cutlass::float_e2m1_t(weight));
        fill_k64_pair(block_b_low, layout_b_low, options.n, low_destination,
                      cutlass::float_e2m1_t(activation));
        ++low_destination;
      }
    }
    block_a_low.sync_device();
    block_a_high.sync_device();
    block_b_low.sync_device();
    block_b_high.sync_device();
    std::cout << "PACK mask_granularity=K64 high_pair_ids=";
    for (size_t i = 0; i < high_pair_ids.size(); ++i) {
      std::cout << (i == 0 ? "" : ",") << high_pair_ids[i];
    }
    std::cout << " low_pairs=" << low_destination
              << " high_pairs=" << high_destination << '\n';
    expected = expected_low + expected_high;
  }

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, 1},
      {
          {block_a_low.device_data(), stride_a_low,
           block_b_low.device_data(), stride_b_low,
           block_sfa_low.device_data(), layout_sfa_low,
           block_sfb_low.device_data(), layout_sfb_low},
          {block_a_high.device_data(), stride_a_high,
           block_b_high.device_data(), stride_b_high,
           block_sfa_high.device_data(), layout_sfa_high,
           block_sfb_high.device_data(), layout_sfb_high},
          options.low_k, high_k},
      {{1.0f, 0.0f},
       block_c.device_data(), stride_c,
       block_d.device_data(), stride_d}};

  Gemm gemm;
  std::cout << "INFO milestone=" << (CompactStreams ? "2b" : "2a")
            << " kernel_smem_bytes=" << sizeof(typename Gemm::GemmKernel::SharedStorage)
            << '\n';
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "can_implement failed: " << cutlassGetStatusString(status) << '\n';
    return false;
  }
  cutlass::device_memory::allocation<uint8_t> workspace(Gemm::get_workspace_size(arguments));
  HYGEMM_CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  HYGEMM_CUTLASS_CHECK(gemm.run());
  HYGEMM_CUDA_CHECK(cudaDeviceSynchronize());
  block_d.sync_host();

  int mismatches = 0;
  float max_abs_error = 0.0f;
  for (int i = 0; i < block_d.capacity(); ++i) {
    float error = std::abs(block_d.host_data()[i] - expected);
    max_abs_error = std::max(max_abs_error, error);
    mismatches += error == 0.0f ? 0 : 1;
  }

  for (int i = 0; i < options.warmup; ++i) {
    HYGEMM_CUTLASS_CHECK(gemm.run());
  }
  HYGEMM_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{};
  HYGEMM_CUDA_CHECK(cudaEventCreate(&start));
  HYGEMM_CUDA_CHECK(cudaEventCreate(&stop));
  HYGEMM_CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < options.iterations; ++i) {
    HYGEMM_CUTLASS_CHECK(gemm.run());
  }
  HYGEMM_CUDA_CHECK(cudaEventRecord(stop));
  HYGEMM_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  HYGEMM_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  HYGEMM_CUDA_CHECK(cudaEventDestroy(start));
  HYGEMM_CUDA_CHECK(cudaEventDestroy(stop));

  float runtime_ms = elapsed_ms / options.iterations;
  double tflops = 2.0 * options.m * options.n * options.k / (runtime_ms * 1.0e9);
  std::cout << std::fixed << std::setprecision(4)
            << "RESULT milestone=" << (CompactStreams ? "2b" : "2a")
            << " representation=" << (CompactStreams ? "compact" : "full_dual")
            << " disposition=" << (mismatches == 0 ? "passed" : "failed")
            << " m=" << options.m << " n=" << options.n << " k=" << options.k
            << " low_k=" << options.low_k << " high_k=" << high_k
            << " runtime_ms=" << runtime_ms << " tflops=" << tflops
            << " mismatches=" << mismatches
            << " max_abs_error=" << max_abs_error
            << " expected=" << expected
            << " first=" << block_d.host_data()[0] << '\n';

  // Keep stock single-path controls adjacent to the custom collective so a
  // failed hybrid run can distinguish input/scale construction from mainloop
  // composition errors.
  {
    using LowGemm = typename Config::LowGemm;
    typename LowGemm::Arguments low_arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, low_physical_k, 1},
        {block_a_low.device_data(), stride_a_low,
         block_b_low.device_data(), stride_b_low,
         block_sfa_low.device_data(), layout_sfa_low,
         block_sfb_low.device_data(), layout_sfb_low},
        {{1.0f, 0.0f}, block_c.device_data(), stride_c,
         block_d.device_data(), stride_d}};
    LowGemm low_gemm;
    cutlass::device_memory::allocation<uint8_t> low_workspace(
        LowGemm::get_workspace_size(low_arguments));
    HYGEMM_CUTLASS_CHECK(low_gemm.initialize(low_arguments, low_workspace.get()));
    HYGEMM_CUTLASS_CHECK(low_gemm.run());
    HYGEMM_CUDA_CHECK(cudaDeviceSynchronize());
    block_d.sync_host();
    std::cout << "CONTROL path=stock_low expected=" << expected_low
              << " first=" << block_d.host_data()[0] << '\n';
  }
  {
    using HighGemm = typename Config::HighGemm;
    typename HighGemm::Arguments high_arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, high_physical_k, 1},
        {block_a_high.device_data(), stride_a_high,
         block_b_high.device_data(), stride_b_high,
         block_sfa_high.device_data(), layout_sfa_high,
         block_sfb_high.device_data(), layout_sfb_high},
        {{1.0f, 0.0f}, block_c.device_data(), stride_c,
         block_d.device_data(), stride_d}};
    HighGemm high_gemm;
    cutlass::device_memory::allocation<uint8_t> high_workspace(
        HighGemm::get_workspace_size(high_arguments));
    HYGEMM_CUTLASS_CHECK(high_gemm.initialize(high_arguments, high_workspace.get()));
    HYGEMM_CUTLASS_CHECK(high_gemm.run());
    HYGEMM_CUDA_CHECK(cudaDeviceSynchronize());
    block_d.sync_host();
    std::cout << "CONTROL path=stock_high expected=" << expected_high
              << " first=" << block_d.host_data()[0] << '\n';
  }
  return mismatches == 0;
}

}  // namespace hygemm

int main(int argc, char const** argv) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  std::cerr << "Binary was not compiled with SM120 MMA support.\n";
  return 1;
#else
  hygemm::Options options;
  options.parse(argc, argv);
  if (!options.valid()) {
    std::cerr << "Usage: --m=<int> --n=<int> --k=<multiple-of-128> "
                 "--low-k=<multiple-of-128> --mode=both|2a|2b "
                 "--warmup=<int> --iterations=<int>\n";
    return 1;
  }

  cudaDeviceProp props{};
  HYGEMM_CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  if (props.major != 12) {
    std::cerr << "SM120 GPU required.\n";
    return 1;
  }
  std::cout << "CONFIG gpu=" << props.name << " mode=" << options.mode << '\n';
  bool passed = true;
  if (options.mode == "both" || options.mode == "2a") {
    passed &= hygemm::run_hybrid<false>(options);
  }
  if (options.mode == "both" || options.mode == "2b") {
    passed &= hygemm::run_hybrid<true>(options);
  }
  return passed ? 0 : 2;
#endif
}
