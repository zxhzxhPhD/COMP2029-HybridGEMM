#include "output_channel_gemm.h"
#include <stdexcept>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#define OC_CAT_IMPL(a,b) a##b
#define OC_CAT(a,b) OC_CAT_IMPL(a,b)
namespace hygemm::oc {
// Macro-configured translation units must not share template/inline symbols.
namespace OC_CAT(variant_,HYGEMM_OC_VARIANT_ID) {
using namespace cute;

// Custom EVT leaf: consume accumulator fragments and scatter directly to final
// output. Void C/D disables the stock epilogue's TMA load/store and output SMEM.
// No intermediate global output, output accumulation or extra scatter kernel.
struct ScatterStore {
  using ElementAux = cutlass::half_t;
  struct SharedStorage {};
  struct Arguments {
    cutlass::half_t* output = nullptr;
    int32_t const* channel_ids = nullptr;
    int output_channels = 0;
  };
  using Params = Arguments;
  Params const* params;
  template<class Shape> static Params to_underlying_arguments(Shape const&, Arguments const& a, void*) { return a; }
  template<class Shape> static bool can_implement(Shape const&, Arguments const& a) {
    return a.output && a.channel_ids && a.output_channels > 0;
  }
  template<class Shape> static size_t get_workspace_size(Shape const&, Arguments const&) { return 0; }
  template<class Shape> static cutlass::Status initialize_workspace(Shape const&, Arguments const&, void*, cudaStream_t,
                                                                  cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }
  CUTLASS_HOST_DEVICE ScatterStore() = default;
  CUTLASS_HOST_DEVICE ScatterStore(Params const& p, SharedStorage const&) : params(&p) {}
  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }
  template<class... T> CUTLASS_DEVICE auto get_producer_load_callbacks(
      cutlass::epilogue::fusion::ProducerLoadArgs<T...> const&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }
  template<class Coordinates> struct Consumer : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    Coordinates coords;
    Params const* params;
    int rows, tokens;
    CUTLASS_DEVICE Consumer(Coordinates c, Params const* p, int m, int n)
      : coords(c), params(p), rows(m), tokens(n) {}
    template<class Acc, int F> CUTLASS_DEVICE auto visit(cutlass::Array<Acc,F> const& acc,
                                                       int epi_v, int epi_m, int epi_n) {
      auto c = coalesce(coords(_,_,_,epi_m,epi_n));
      CUTLASS_PRAGMA_UNROLL
      for (int i=0;i<F;++i) {
        auto mn = c(epi_v*F+i);
        int row = get<0>(mn), token = get<1>(mn);
        if (row < rows && token < tokens) {
          int channel = params->channel_ids[row];
          params->output[int64_t(token)*params->output_channels+channel] = cutlass::half_t(acc[i]);
        }
      }
      return acc;
    }
  };
  template<bool ReferenceSrc, class... T> CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::ConsumerStoreArgs<T...> const& args) {
    auto [m,n,k,l] = args.problem_shape_mnkl;
    auto identity = make_identity_tensor(make_shape(m,n,l));
    auto coords = cutlass::epilogue::fusion::detail::sm90_partition_for_epilogue<ReferenceSrc>(
      identity,args.tile_shape_mnk,args.tile_coord_mnkl,args.epi_tile,args.tiled_copy,args.thread_idx);
    return Consumer<decltype(coords)>(coords,params,m,n);
  }
};

template<bool FP8> struct Configuration {
  using Weight = cute::conditional_t<FP8,cutlass::mx_float8_t<cutlass::float_e4m3_t>,
                                       cutlass::mx_float4_t<cutlass::float_e2m1_t>>;
  using Activation = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using Tile = Shape<_128,Int<HYGEMM_OC_TILE_N>,Int<HYGEMM_OC_TILE_K>>;
  using Cluster = Shape<_1,_1,_1>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,cutlass::arch::OpClassTensorOp,Tile,Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,float,float,
    void,cutlass::layout::ColumnMajor,8,void,cutlass::layout::ColumnMajor,8,
    cutlass::epilogue::collective::EpilogueScheduleAuto,ScatterStore>::CollectiveOp;
  using Stages = cute::conditional_t<HYGEMM_OC_STAGES == 0,
    cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::StageCount<HYGEMM_OC_STAGES>>;
  using Schedule = cute::conditional_t<HYGEMM_OC_PINGPONG,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,cutlass::gemm::collective::KernelScheduleAuto>;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,cutlass::arch::OpClassBlockScaledTensorOp,
    Weight,cutlass::layout::RowMajor,128/cutlass::sizeof_bits<typename Weight::DataType>::value,
    Activation,cutlass::layout::ColumnMajor,16,float,Tile,Cluster,Stages,Schedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

inline void check(cutlass::Status s) {
  if (s != cutlass::Status::kSuccess) throw std::runtime_error(cutlassGetStatusString(s));
}
template<bool FP8> class Instance final : public Kernel {
  using C = Configuration<FP8>;
  using GK = typename C::GemmKernel;
  using Gemm = typename C::Gemm;
  Gemm gemm_;
  cutlass::device_memory::allocation<uint8_t> workspace_;
  KernelInfo info_;
 public:
  Instance(DeviceProblem const& p,cudaStream_t stream) {
    if (p.rows<=0 || p.tokens<=0 || p.input_channels<=0 || p.input_channels%128 ||
        p.rows>p.output_channels || !p.weight || !p.activation || !p.weight_scales || !p.activation_scales ||
        !p.channel_ids || !p.output) throw std::runtime_error("Invalid output-channel GEMM problem");
    auto sa=cutlass::make_cute_packed_stride(typename GK::StrideA{},{p.rows,p.input_channels,1});
    auto sb=cutlass::make_cute_packed_stride(typename GK::StrideB{},{p.tokens,p.input_channels,1});
    auto sd=cutlass::make_cute_packed_stride(typename GK::StrideD{},{p.rows,p.tokens,1});
    auto sc=cutlass::make_cute_packed_stride(typename GK::StrideC{},{p.rows,p.tokens,1});
    using BS=typename C::Mainloop::Sm1xxBlkScaledConfig;
    auto shape=make_shape(p.rows,p.tokens,p.input_channels,1);
    auto la=BS::tile_atom_to_shape_SFA(shape),lb=BS::tile_atom_to_shape_SFB(shape);
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
      {p.rows,p.tokens,p.input_channels,1},
      {static_cast<typename C::Weight::DataType const*>(p.weight),sa,
       static_cast<cutlass::float_e4m3_t const*>(p.activation),sb,
       static_cast<cutlass::float_ue8m0_t const*>(p.weight_scales),la,
       static_cast<cutlass::float_ue8m0_t const*>(p.activation_scales),lb},
      {{static_cast<cutlass::half_t*>(p.output),p.channel_ids,p.output_channels},nullptr,sc,nullptr,sd}};
    check(gemm_.can_implement(args));
    workspace_.reset(Gemm::get_workspace_size(args));
    check(gemm_.initialize(args,workspace_.get(),stream));
    cudaFuncAttributes attr{};
    auto status=cudaFuncGetAttributes(&attr,cutlass::device_kernel<GK>);
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    info_={HYGEMM_OC_VARIANT_ID,128,HYGEMM_OC_TILE_N,HYGEMM_OC_TILE_K,C::Mainloop::DispatchPolicy::Stages,
           attr.numRegs,int(sizeof(typename GK::SharedStorage)),int(attr.localSizeBytes),FP8,bool(HYGEMM_OC_PINGPONG)};
  }
  void run(cudaStream_t stream) override { check(gemm_.run(stream)); }
  KernelInfo info() const override { return info_; }
};

} // variant-specific namespace
std::unique_ptr<Kernel> OC_CAT(make_variant_,HYGEMM_OC_VARIANT_ID)(bool fp8,DeviceProblem const&p,cudaStream_t stream) {
  if(fp8) return std::make_unique<OC_CAT(variant_,HYGEMM_OC_VARIANT_ID)::Instance<true>>(p,stream);
  return std::make_unique<OC_CAT(variant_,HYGEMM_OC_VARIANT_ID)::Instance<false>>(p,stream);
}
} // namespace hygemm::oc
