#include "output_channel_fused.h"
#include "output_channel_switch_collective.cuh"
#include <stdexcept>
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
namespace OC_CAT(fused_variant_,HYGEMM_OC_VARIANT_ID) {
using namespace cute;

struct ScatterStore {
  using ElementAux=cutlass::half_t;
  struct SharedStorage {};
  struct Arguments {
    cutlass::half_t* output;
    int32_t const* low_ids;
    int32_t const* high_ids;
    int low_rows,high_rows,low_virtual_rows,output_channels;
  };
  using Params=Arguments;Params const* params;
  template<class Shape> static Params to_underlying_arguments(Shape const&,Arguments const&a,void*) {return a;}
  template<class Shape> static bool can_implement(Shape const&,Arguments const&a) {
    return a.output && (!a.low_rows || a.low_ids) && (!a.high_rows || a.high_ids) && a.output_channels>0;
  }
  template<class Shape> static size_t get_workspace_size(Shape const&,Arguments const&) {return 0;}
  template<class Shape> static cutlass::Status initialize_workspace(Shape const&,Arguments const&,void*,cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) {return cutlass::Status::kSuccess;}
  CUTLASS_HOST_DEVICE ScatterStore() {}
  CUTLASS_HOST_DEVICE ScatterStore(Params const&p,SharedStorage const&):params(&p) {}
  CUTLASS_DEVICE bool is_producer_load_needed() const {return false;}
  CUTLASS_DEVICE bool is_C_load_needed() const {return false;}
  template<class...T> CUTLASS_DEVICE auto get_producer_load_callbacks(cutlass::epilogue::fusion::ProducerLoadArgs<T...> const&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }
  template<class Coordinates> struct Consumer:cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    Coordinates coords;cutlass::half_t* output;int32_t const* ids;
    int row_base,rows,tokens,output_channels;
    CUTLASS_DEVICE Consumer(Coordinates c,Params const*p,int n,bool high)
      :coords(c),output(p->output),ids(high?p->high_ids:p->low_ids),
       row_base(high?p->low_virtual_rows:0),rows(high?p->high_rows:p->low_rows),tokens(n),output_channels(p->output_channels) {}
    template<class A,int F> CUTLASS_DEVICE auto visit(cutlass::Array<A,F> const&acc,int v,int m,int n) {
      auto c=coalesce(coords(_,_,_,m,n));
      CUTLASS_PRAGMA_UNROLL
      for(int i=0;i<F;++i) {
        auto mn=c(v*F+i);int row=get<0>(mn)-row_base,token=get<1>(mn);
        if(row<rows && token<tokens) {
          int ch=ids[row];
          output[int64_t(token)*output_channels+ch]=cutlass::half_t(acc[i]);
        }
      }
      return acc;
    }
  };
  template<bool Ref,class...T> CUTLASS_DEVICE auto get_consumer_store_callbacks(cutlass::epilogue::fusion::ConsumerStoreArgs<T...> const&a) {
    auto [m,n,k,l]=a.problem_shape_mnkl;
    auto coords=cutlass::epilogue::fusion::detail::sm90_partition_for_epilogue<Ref>(make_identity_tensor(make_shape(m,n,l)),
      a.tile_shape_mnk,a.tile_coord_mnkl,a.epi_tile,a.tiled_copy,a.thread_idx);
    // Precision is uniform for the entire output CTA, including all epilogue
    // subtiles. Hoist map selection out of the per-element store path.
    bool high=int(get<0>(a.tile_coord_mnkl))*int(get<0>(a.tile_shape_mnk))>=params->low_virtual_rows;
    return Consumer<decltype(coords)>(coords,params,n,high);
  }
};
using Tile=Shape<_128,Int<HYGEMM_OC_TILE_N>,Int<HYGEMM_OC_TILE_K>>;
using Cluster=Shape<_1,_1,_1>;
using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm120,cutlass::arch::OpClassTensorOp,Tile,Cluster,
  cutlass::epilogue::collective::EpilogueTileAuto,float,float,
  void,cutlass::layout::ColumnMajor,8,void,cutlass::layout::ColumnMajor,8,
  cutlass::epilogue::collective::EpilogueScheduleAuto,ScatterStore>::CollectiveOp;
using Stages=cute::conditional_t<HYGEMM_OC_STAGES==0,
  cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epilogue::SharedStorage))>,
  cutlass::gemm::collective::StageCount<HYGEMM_OC_STAGES>>;
using Schedule=cute::conditional_t<HYGEMM_OC_PINGPONG,cutlass::gemm::KernelTmaWarpSpecializedPingpong,
  cutlass::gemm::collective::KernelScheduleAuto>;
template<class W> using Stock=typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm120,cutlass::arch::OpClassBlockScaledTensorOp,
  W,cutlass::layout::RowMajor,128/cutlass::sizeof_bits<typename W::DataType>::value,
  cutlass::mx_float8_t<cutlass::float_e4m3_t>,cutlass::layout::ColumnMajor,16,
  float,Tile,Cluster,Stages,Schedule>::CollectiveOp;
using Low=Stock<cutlass::mx_float4_t<cutlass::float_e2m1_t>>;
using High=Stock<cutlass::mx_float8_t<cutlass::float_e4m3_t>>;
using Mainloop=OutputChannelSwitchCollective<Low,High>;
using GK=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue>;
using Gemm=cutlass::gemm::device::GemmUniversalAdapter<GK>;

inline void check(cutlass::Status s) {if(s!=cutlass::Status::kSuccess)throw std::runtime_error(cutlassGetStatusString(s));}
template<class C> auto partition_args(DeviceProblem const&p,DeviceProblem const&fallback) {
  int rows=p.rows?p.rows:1;
  auto a=cutlass::make_cute_packed_stride(typename C::StrideA{},{rows,p.input_channels,1});
  auto b=cutlass::make_cute_packed_stride(typename C::StrideB{},{p.tokens,p.input_channels,1});
  using BS=typename C::Sm1xxBlkScaledConfig;
  auto shape=make_shape(rows,p.tokens,p.input_channels,1);
  return typename C::Arguments{static_cast<typename C::ElementA const*>(p.rows?p.weight:fallback.weight),a,
    static_cast<typename C::ElementB const*>(p.activation),b,
    static_cast<typename C::ElementSF const*>(p.rows?p.weight_scales:fallback.weight_scales),BS::tile_atom_to_shape_SFA(shape),
    static_cast<typename C::ElementSF const*>(p.activation_scales),BS::tile_atom_to_shape_SFB(shape)};
}
class Instance final:public FusedKernel {
  Gemm gemm;
  cutlass::device_memory::allocation<uint8_t> workspace;
  FusedInfo metadata;
 public:
  Instance(DeviceProblem const&l,DeviceProblem const&h,cudaStream_t stream,uint32_t*audit) {
    if(l.rows<0 || h.rows<0 || l.rows+h.rows!=l.output_channels || l.output_channels<=0 ||
       l.output_channels!=h.output_channels || l.tokens<=0 || l.tokens!=h.tokens || l.input_channels<=0 ||
       l.input_channels%128 || l.input_channels!=h.input_channels || l.activation!=h.activation ||
       l.activation_scales!=h.activation_scales || l.output!=h.output || !l.output || !l.activation || !l.activation_scales ||
       (l.rows && (!l.weight || !l.weight_scales || !l.channel_ids)) ||
       (h.rows && (!h.weight || !h.weight_scales || !h.channel_ids))) throw std::runtime_error("Invalid fused partition contract");
    int low_virtual=(l.rows+127)/128*128,virtual_rows=low_virtual+(h.rows+127)/128*128;
    auto sc=cutlass::make_cute_packed_stride(typename GK::StrideC{},{virtual_rows,l.tokens,1});
    auto sd=cutlass::make_cute_packed_stride(typename GK::StrideD{},{virtual_rows,l.tokens,1});
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{virtual_rows,l.tokens,l.input_channels,1},
      {partition_args<Low>(l,h),partition_args<High>(h,l),l.rows,h.rows,audit},
      {{static_cast<cutlass::half_t*>(l.output),l.channel_ids,h.channel_ids,l.rows,h.rows,low_virtual,l.output_channels},nullptr,sc,nullptr,sd}};
    check(gemm.can_implement(args));workspace.reset(Gemm::get_workspace_size(args));check(gemm.initialize(args,workspace.get(),stream));
    cudaFuncAttributes a{};auto status=cudaFuncGetAttributes(&a,cutlass::device_kernel<GK>);
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    metadata={{HYGEMM_OC_VARIANT_ID,128,HYGEMM_OC_TILE_N,HYGEMM_OC_TILE_K,Mainloop::DispatchPolicy::Stages,
      a.numRegs,int(sizeof(typename GK::SharedStorage)),int(a.localSizeBytes),false,bool(HYGEMM_OC_PINGPONG)},
      Mainloop::LowBytes,Mainloop::HighBytes,virtual_rows};
  }
  void run(cudaStream_t stream) override {check(gemm.run(stream));}
  KernelInfo info() const override {return metadata.kernel;}
  FusedInfo fused_info() const override {return metadata;}
};
}
std::unique_ptr<FusedKernel> OC_CAT(make_fused_variant_,HYGEMM_OC_VARIANT_ID)(DeviceProblem const&l,DeviceProblem const&h,cudaStream_t s,uint32_t*a) {
  return std::make_unique<OC_CAT(fused_variant_,HYGEMM_OC_VARIANT_ID)::Instance>(l,h,s,a);
}
}
