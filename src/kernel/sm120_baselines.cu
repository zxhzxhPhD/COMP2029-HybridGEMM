#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

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
  int m = 2048;
  int n = 174;
  int k = 2048;
  int warmup = 5;
  int iterations = 20;
  uint64_t seed = 20260712;
  std::string kernel = "both";
  bool help = false;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    help = cmd.check_cmd_line_flag("help");
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("seed", seed);
    cmd.get_cmd_line_argument("kernel", kernel);
  }

  bool valid() const {
    return m > 0 && n > 0 && k > 0 && warmup >= 0 && iterations > 0 &&
           (kernel == "both" || kernel == "w4a4" || kernel == "w4a8");
  }

  double tflops(double runtime_ms) const {
    return (2.0 * double(m) * double(n) * double(k)) / (runtime_ms * 1.0e9);
  }

  static void usage() {
    std::cout
        << "SM120 MX block-scaled GEMM baselines\n\n"
        << "  --m=<int> --n=<int> --k=<int>\n"
        << "  --kernel=both|w4a4|w4a8\n"
        << "  --warmup=<int> --iterations=<int> --seed=<int>\n";
  }
};

template <class ElementB_>
struct BaselineConfig {
  using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  using ElementB = ElementB_;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementC = float;
  using ElementD = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  // (M,N) column-major is equivalent to the usual token-major [N,M]
  // activation storage and keeps the contiguous M dimension aligned for any N.
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<typename ElementA::DataType>::value;
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<typename ElementB::DataType>::value;
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

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
          ElementA, LayoutA, AlignmentA,
          ElementB, LayoutB, AlignmentB,
          ElementAccumulator,
          TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using W4A4Config = BaselineConfig<cutlass::mx_float4_t<cutlass::float_e2m1_t>>;
using W4A8Config = BaselineConfig<cutlass::mx_float8_t<cutlass::float_e4m3_t>>;

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

template <typename Element, typename Layout>
void fill_random(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
  double max = 1.0;
  double min = -1.0;
  if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
    max = 4.0;
    min = 1.0;
  }
  cutlass::reference::host::TensorFillRandomUniform(view, seed, max, min, 0);
}

struct RunResult {
  bool passed = false;
  double runtime_ms = 0.0;
  double tflops = 0.0;
};

template <class Config>
RunResult run_baseline(Options const& options, char const* name) {
  using Gemm = typename Config::Gemm;
  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementC = typename Config::ElementC;
  using ElementD = typename Config::ElementD;
  using ElementAccumulator = typename Config::ElementAccumulator;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutA = decltype(make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutB = decltype(make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutC = decltype(make_layout(make_shape(0, 0, 0), StrideC{}));
  using LayoutD = decltype(make_layout(make_shape(0, 0, 0), StrideD{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using BlockScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  StrideA stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, {options.m, options.k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, {options.n, options.k, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(
      StrideC{}, {options.m, options.n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, {options.m, options.n, 1});

  LayoutA layout_a = make_layout(make_shape(options.m, options.k, 1), stride_a);
  LayoutB layout_b = make_layout(make_shape(options.n, options.k, 1), stride_b);
  LayoutC layout_c = make_layout(make_shape(options.m, options.n, 1), stride_c);
  LayoutD layout_d = make_layout(make_shape(options.m, options.n, 1), stride_d);
  LayoutSFA layout_sfa = BlockScaledConfig::tile_atom_to_shape_SFA(
      make_shape(options.m, options.n, options.k, 1));
  LayoutSFB layout_sfb = BlockScaledConfig::tile_atom_to_shape_SFB(
      make_shape(options.m, options.n, options.k, 1));

  cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout>
      block_a(cutlass::make_Coord(size(layout_a)));
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout>
      block_b(cutlass::make_Coord(size(layout_b)));
  cutlass::HostTensor<typename ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout>
      block_sfa(cutlass::make_Coord(size(filter_zeros(layout_sfa))));
  cutlass::HostTensor<typename ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout>
      block_sfb(cutlass::make_Coord(size(filter_zeros(layout_sfb))));
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout>
      block_c(cutlass::make_Coord(size(layout_c)));
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout>
      block_d(cutlass::make_Coord(size(layout_d)));
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout>
      reference_d(cutlass::make_Coord(size(layout_d)));

  fill_random(block_a.host_view(), options.seed + 1);
  fill_random(block_b.host_view(), options.seed + 2);
  fill_random(block_sfa.host_view(), options.seed + 3);
  fill_random(block_sfb.host_view(), options.seed + 4);
  fill_random(block_c.host_view(), options.seed + 5);
  block_a.sync_device();
  block_b.sync_device();
  block_sfa.sync_device();
  block_sfb.sync_device();
  block_c.sync_device();

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, 1},
      {block_a.device_data(), stride_a,
       block_b.device_data(), stride_b,
       block_sfa.device_data(), layout_sfa,
       block_sfb.device_data(), layout_sfb},
      {{1.0f, 0.0f},
       block_c.device_data(), stride_c,
       block_d.device_data(), stride_d}};

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << name << ": can_implement failed: " << cutlassGetStatusString(status) << '\n';
    return {};
  }
  cutlass::device_memory::allocation<uint8_t> workspace(Gemm::get_workspace_size(arguments));
  HYGEMM_CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  HYGEMM_CUTLASS_CHECK(gemm.run());
  HYGEMM_CUDA_CHECK(cudaDeviceSynchronize());

  auto tensor_a = make_tensor(make_iterator(block_a.host_data()), layout_a);
  auto tensor_b = make_tensor(make_iterator(block_b.host_data()), layout_b);
  auto tensor_sfa = make_tensor(block_sfa.host_data(), layout_sfa);
  auto tensor_sfb = make_tensor(block_sfb.host_data(), layout_sfb);
  auto tensor_c = make_tensor(make_iterator(block_c.host_data()), layout_c);
  auto tensor_ref_d = make_tensor(make_iterator(reference_d.host_data()), layout_d);

  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator, decltype(tensor_a), decltype(tensor_sfa),
      decltype(tensor_b), decltype(tensor_sfb)>
      mainloop_params{tensor_a, tensor_sfa, tensor_b, tensor_sfb};
  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAccumulator, ElementAccumulator, ElementAccumulator,
      decltype(tensor_c), decltype(tensor_ref_d)>
      epilogue_params{1.0f, 0.0f, tensor_c, tensor_ref_d};
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  block_d.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(
      reference_d.host_view(), block_d.host_view());
  passed = passed && cutlass::reference::host::TensorNorm(block_d.host_view()) > 0;

  for (int i = 0; i < options.warmup; ++i) {
    HYGEMM_CUTLASS_CHECK(gemm.run());
  }
  HYGEMM_CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
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

  RunResult result;
  result.passed = passed;
  result.runtime_ms = elapsed_ms / options.iterations;
  result.tflops = options.tflops(result.runtime_ms);
  std::cout << "RESULT kernel=" << name
            << " disposition=" << (passed ? "passed" : "failed")
            << " m=" << options.m << " n=" << options.n << " k=" << options.k
            << " runtime_ms=" << result.runtime_ms
            << " tflops=" << result.tflops << '\n';
  return result;
}

}  // namespace hygemm

int main(int argc, char const** argv) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  std::cerr << "Binary was not compiled with SM120 MMA support.\n";
  return 1;
#else
  hygemm::Options options;
  options.parse(argc, argv);
  if (options.help) {
    hygemm::Options::usage();
    return 0;
  }
  if (!options.valid()) {
    hygemm::Options::usage();
    return 1;
  }

  int device = 0;
  cudaDeviceProp props{};
  HYGEMM_CUDA_CHECK(cudaGetDevice(&device));
  HYGEMM_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  if (props.major != 12) {
    std::cerr << "SM120 GPU required; found compute capability "
              << props.major << '.' << props.minor << '\n';
    return 1;
  }

  bool passed = true;
  if (options.kernel == "both" || options.kernel == "w4a4") {
    passed &= hygemm::run_baseline<hygemm::W4A4Config>(options, "w4a4_k64").passed;
  }
  if (options.kernel == "both" || options.kernel == "w4a8") {
    passed &= hygemm::run_baseline<hygemm::W4A8Config>(options, "w4a8").passed;
  }
  return passed ? 0 : 2;
#endif
}
