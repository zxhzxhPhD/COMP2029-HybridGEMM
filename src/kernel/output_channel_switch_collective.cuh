#pragma once
#include <type_traits>
#include <cstddef>
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

namespace hygemm::oc {

// Keep the stock SM120 kernel/scheduler and one pipeline. A virtual M dimension
// concatenates whole low/high output tiles; each tile selects its own MMA.
template<class Low, class High>
struct OutputChannelSwitchCollective : Low {
  using typename Low::TileShape;
  using StockPipeline = typename Low::MainloopPipeline;
  // All views reference the SAME barrier arrays and retain their ring phase.
  // Only the producer's per-tile expected byte count changes. Constructing the
  // view with InitBarriers=false must never reinitialize an in-flight stage.
  struct MainloopPipeline : StockPipeline {
    using Params = typename StockPipeline::Params;
    using SharedStorage = typename StockPipeline::SharedStorage;
    SharedStorage* storage;
    Params original_params;
    template<class Cluster> CUTLASS_DEVICE MainloopPipeline(SharedStorage&s,Params p,Cluster c)
      : StockPipeline(s,p,c),storage(&s),original_params(p) {}
    CUTLASS_DEVICE StockPipeline for_transaction_bytes(uint32_t bytes) const {
      auto p=original_params;p.transaction_bytes=bytes;
      return StockPipeline(*storage,p,cute::Shape<cute::_1,cute::_1,cute::_1>{},cute::false_type{},cute::true_type{});
    }
  };
  using typename Low::PipelineState;
  using typename Low::PipelineStorage;
  static constexpr int TileM = cute::size<0>(TileShape{});
  static constexpr uint32_t LowBytes = Low::TmaTransactionBytes;
  static constexpr uint32_t HighBytes = High::TmaTransactionBytes;
  static_assert(HighBytes > LowBytes && LowBytes > 0);
  static_assert(cute::is_same_v<typename Low::DispatchPolicy,typename High::DispatchPolicy>);
  static_assert(cute::is_same_v<typename Low::TileShape,typename High::TileShape>);
  static_assert(cute::is_same_v<typename Low::MainloopPipeline,typename High::MainloopPipeline>);
  static_assert(cute::is_same_v<typename Low::SmemLayoutA,typename High::SmemLayoutA>);
  static_assert(cute::is_same_v<typename Low::SmemLayoutB,typename High::SmemLayoutB>);
  static_assert(cute::is_same_v<typename Low::SmemLayoutSFA,typename High::SmemLayoutSFA>);
  static_assert(cute::is_same_v<typename Low::SmemLayoutSFB,typename High::SmemLayoutSFB>);
  using LowFragment = decltype(cute::partition_fragment_C(typename Low::TiledMma{},cute::take<0,2>(TileShape{})));
  using HighFragment = decltype(cute::partition_fragment_C(typename High::TiledMma{},cute::take<0,2>(TileShape{})));
  static_assert(cute::is_same_v<LowFragment,HighFragment>);

  using LS = typename Low::TensorStorage;
  using HS = typename High::TensorStorage;
  // All members form the same standard-layout common initial sequence. The
  // union overlays one allocation, including stages that can belong to
  // different output tiles simultaneously. It is NOT two precision buffers.
  static_assert(std::is_standard_layout_v<LS> && std::is_standard_layout_v<HS>);
#define OC_CHECK_MEMBER(member) \
  static_assert(std::is_same_v<decltype(LS::member),decltype(HS::member)>); \
  static_assert(offsetof(LS,member)==offsetof(HS,member));
  OC_CHECK_MEMBER(smem_A) OC_CHECK_MEMBER(smem_B)
  OC_CHECK_MEMBER(smem_SFA) OC_CHECK_MEMBER(smem_SFB)
#undef OC_CHECK_MEMBER
  static_assert(sizeof(LS)==sizeof(HS) && alignof(LS)==alignof(HS));
  union TensorStorage {
    LS low;
    HS high;
    CUTLASS_HOST_DEVICE TensorStorage() {}
  };
  struct Arguments {
    typename Low::Arguments low;
    typename High::Arguments high;
    int low_rows, high_rows;
    uint32_t* audit = nullptr;
  };
  struct Params {
    typename Low::Params low;
    typename High::Params high;
    int low_rows, high_rows, low_tiles;
    uint32_t* audit;
    // The outer kernel's initial view uses this nonzero default. Producer
    // tile views below override it with the selected path's exact byte count.
    uint32_t tma_transaction_bytes = LowBytes;
  };
  int last_load_precision = -1;
  int last_mma_precision = -1;

  template<class Shape> CUTLASS_HOST_DEVICE static auto partition_shape(Shape shape,int rows) {
    cute::get<0>(shape)=rows>0?rows:1; return shape;
  }
  template<class Shape> static Params to_underlying_arguments(Shape const& shape,Arguments const&a,void*w) {
    return {Low::to_underlying_arguments(partition_shape(shape,a.low_rows),a.low,w),
      High::to_underlying_arguments(partition_shape(shape,a.high_rows),a.high,w),
      a.low_rows,a.high_rows,(a.low_rows+TileM-1)/TileM,a.audit,LowBytes};
  }
  template<class Shape> static bool can_implement(Shape const&s,Arguments const&a) {
    return a.low_rows>=0 && a.high_rows>=0 && a.low_rows+a.high_rows>0 &&
      (!a.low_rows || Low::can_implement(partition_shape(s,a.low_rows),a.low)) &&
      (!a.high_rows || High::can_implement(partition_shape(s,a.high_rows),a.high));
  }
  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const&p) {
    if(p.low_rows)Low::prefetch_tma_descriptors(p.low);
    if(p.high_rows)High::prefetch_tma_descriptors(p.high);
  }
  template<class Shape> CUTLASS_DEVICE auto load_init(Shape const&s,Params const&p) const {
    using namespace cute;
    auto low = Low{}.load_init(partition_shape(s,p.low_rows),p.low);
    auto high = High{}.load_init(partition_shape(s,p.high_rows),p.high);
    auto [m,n,k,l]=s;
    auto virtual_a=local_tile(make_identity_tensor(make_shape(m,k,l)),TileShape{},
                             make_coord(_,_,_),Step<_1,Underscore,_1>{});
    // The outer kernel only reads shapes of the first two tensors for work
    // coordinates. Actual TMA copies use partition-local descriptors/tensors.
    return make_tuple(virtual_a,get<1>(low),low,high);
  }

  template<class C,class Inputs,class Coord,class Iter,class Storage>
  CUTLASS_DEVICE static void load_partition(typename C::Params const&p,MainloopPipeline pipeline,
      PipelineState state,Inputs const& inputs,Coord const& coord,Iter kt,int count,Storage& storage) {
    using namespace cute;
    auto selected_pipeline=pipeline.for_transaction_bytes(C::TmaTransactionBytes);
    if(elect_one_sync()) {
      auto sA=make_tensor(make_smem_ptr(storage.smem_A.begin()),typename C::SmemLayoutA{});
      auto sB=make_tensor(make_smem_ptr(storage.smem_B.begin()),typename C::SmemLayoutB{});
      auto sFA=make_tensor(make_smem_ptr(storage.smem_SFA.begin()),typename C::SmemLayoutSFA{});
      auto sFB=make_tensor(make_smem_ptr(storage.smem_SFB.begin()),typename C::SmemLayoutSFB{});
      auto [a,b,fa,fb]=inputs;
      auto [m,n,k,l]=coord;
      auto broadcast_n=make_layout(make_shape(Int<size<1>(typename C::TileShapeSFB{})/size<1>(TileShape{})>{},
                                     Int<cute::numeric_limits<int>::max()>{}),
                                  make_stride(_0{},size<1>(typename C::TileShapeSFB{})/size<1>(TileShape{})));
      auto ca=p.tma_load_a.get_slice(0);auto cb=p.tma_load_b.get_slice(0);
      auto cf=p.tma_load_sfa.get_slice(0);auto cx=p.tma_load_sfb.get_slice(0);
      auto ga=ca.partition_S(a(_,_,m,_,l));auto sa=ca.partition_D(sA);
      auto gb=cb.partition_S(b(_,_,n,_,l));auto sb=cb.partition_D(sB);
      auto gf=cf.partition_S(fa(_,_,m,_,l));auto sf=cf.partition_D(sFA);
      auto gx=cx.partition_S(fb(_,_,broadcast_n(n),_,l));auto sx=cx.partition_D(sFB);
      CUTLASS_PRAGMA_NO_UNROLL
      for(;count>0;--count,++kt,++state) {
        selected_pipeline.producer_acquire(state);
        auto* barrier=selected_pipeline.producer_get_barrier(state);
        int stage=state.index();
        copy(p.tma_load_a.with(*barrier),ga(_,_,_,*kt),sa(_,_,_,stage));
        copy(p.tma_load_b.with(*barrier),gb(_,_,_,*kt),sb(_,_,_,stage));
        copy(p.tma_load_sfa.with(*barrier),gf(_,_,_,*kt),sf(_,_,_,stage));
        copy(p.tma_load_sfb.with(*barrier),gx(_,_,_,*kt),sx(_,_,_,stage));
      }
    }
    __syncwarp();
  }
  template<class Inputs,class Coord,class Iter> CUTLASS_DEVICE void load(Params const&p,
      MainloopPipeline pipeline,PipelineState state,Inputs const& inputs,Coord const& coord,
      Iter kt,int count,int thread,uint32_t,TensorStorage& storage) {
    using namespace cute;
    int m=get<0>(coord);bool high=m>=p.low_tiles;
    auto local=make_coord(high?m-p.low_tiles:m,get<1>(coord),get<2>(coord),get<3>(coord));
    if(p.audit && thread==0) {
      atomicAdd(p.audit+(high?1:0),1u);atomicAdd(p.audit+(high?5:4),uint32_t(count));
      if(last_load_precision>=0 && last_load_precision!=int(high))atomicAdd(p.audit+2,1u);
      last_load_precision=high;
    }
    if(high)load_partition<High>(p.high,pipeline,state,get<3>(inputs),local,kt,count,storage.high);
    else load_partition<Low>(p.low,pipeline,state,get<2>(inputs),local,kt,count,storage.low);
  }
  template<class Acc,class Coord> CUTLASS_DEVICE void mma(MainloopPipeline pipeline,PipelineState state,
      Acc& acc,int count,int thread,TensorStorage& storage,Params const&p,Coord const&coord) {
    using namespace cute;
    int m=get<0>(coord);bool high=m>=p.low_tiles;
    auto local=make_coord(high?m-p.low_tiles:m,get<1>(coord),get<2>(coord),get<3>(coord));
    if(p.audit && thread==0) {
      if(last_mma_precision>=0 && last_mma_precision!=int(high))atomicAdd(p.audit+3,1u);
      last_mma_precision=high;
    }
    // Each stock mma() clears acc once per OUTPUT tile, not per K group.
    if(high)High{}.mma(pipeline,state,acc,count,thread,storage.high,p.high,local);
    else Low{}.mma(pipeline,state,acc,count,thread,storage.low,p.low,local);
  }
  CUTLASS_DEVICE void load_tail(MainloopPipeline p,PipelineState s) {Low{}.load_tail(p,s);}
  CUTLASS_DEVICE void mma_tail(MainloopPipeline p,PipelineState s,int n) {Low{}.mma_tail(p,s,n);}
};
} // namespace hygemm::oc
