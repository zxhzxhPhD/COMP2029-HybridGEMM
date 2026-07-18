#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp"
#include "cute/tensor.hpp"

namespace hygemm {

// Functional K64-paired hybrid wrapper around two stock SM120 block-scaled
// collectives. LowCollective owns the E2M1 x E2M1 K64 OMMA path and
// HighCollective owns the E2M1 x E4M3 K32 QMMA path. The high phase executes
// first, followed by low; both consume one CUTLASS mainloop pipeline and
// accumulate into the same register fragment.
//
// CompactStreams=false (Milestone 2A): both descriptors cover the full K
// tensor. The low stream consumes [0, low_k), and the high stream consumes
// [low_k, K).
//
// CompactStreams=true (Milestone 2B): each descriptor covers its own compact K
// tensor and both streams start at physical K=0. The packer is responsible for
// preserving the logical reduction mapping.
template <class LowCollective_, class HighCollective_, bool CompactStreams>
struct Sm120K64PairedHybridCollective {
  using LowCollective = LowCollective_;
  using HighCollective = HighCollective_;

  using DispatchPolicy = typename LowCollective::DispatchPolicy;
  using TileShape = typename LowCollective::TileShape;
  using ElementA = typename LowCollective::ElementA;
  using ElementB = typename LowCollective::ElementB;
  using ElementSF = typename LowCollective::ElementSF;
  using ElementAccumulator = typename LowCollective::ElementAccumulator;
  using StrideA = typename LowCollective::StrideA;
  using StrideB = typename LowCollective::StrideB;
  using LayoutSFA = typename LowCollective::LayoutSFA;
  using LayoutSFB = typename LowCollective::LayoutSFB;
  using TiledMma = typename LowCollective::TiledMma;
  using ArchTag = typename LowCollective::ArchTag;
  using Sm1xxBlkScaledConfig = typename LowCollective::Sm1xxBlkScaledConfig;

  using MainloopPipeline = typename LowCollective::MainloopPipeline;
  using PipelineState = typename LowCollective::PipelineState;
  using PipelineStorage = typename LowCollective::PipelineStorage;

  static constexpr int NumProducerThreadEvents = LowCollective::NumProducerThreadEvents;
  static constexpr int KTile = cute::size<2>(TileShape{});
  // Initialize every stage for the larger transaction. A low stage retires
  // the byte difference explicitly after issuing its four TMA loads. This
  // avoids changing the expected byte count after producer_acquire(), when a
  // too-small initial transaction can make the barrier visible prematurely.
  static constexpr uint32_t TmaTransactionBytes = HighCollective::TmaTransactionBytes;

  static_assert(cute::is_same_v<typename LowCollective::DispatchPolicy,
                                typename HighCollective::DispatchPolicy>);
  static_assert(cute::is_same_v<typename LowCollective::TileShape,
                                typename HighCollective::TileShape>);
  static_assert(cute::is_same_v<typename LowCollective::ElementAccumulator,
                                typename HighCollective::ElementAccumulator>);
  static_assert(cute::is_same_v<typename LowCollective::MainloopPipeline,
                                typename HighCollective::MainloopPipeline>);
  static_assert(cute::size(typename LowCollective::TiledMma{}) ==
                cute::size(typename HighCollective::TiledMma{}));
  using LowCFragment = decltype(cute::partition_fragment_C(
      typename LowCollective::TiledMma{}, cute::take<0, 2>(TileShape{})));
  using HighCFragment = decltype(cute::partition_fragment_C(
      typename HighCollective::TiledMma{}, cute::take<0, 2>(TileShape{})));
  static_assert(cute::is_same_v<LowCFragment, HighCFragment>);
  static_assert(HighCollective::TmaTransactionBytes >= LowCollective::TmaTransactionBytes);

  // The two typed streams are consumed in disjoint phases. Overlay their SMEM
  // storage and drain the pipeline at the type boundary; this keeps the custom
  // collective within SM120's opt-in shared-memory capacity.
  union TensorStorage {
    typename LowCollective::TensorStorage low;
    typename HighCollective::TensorStorage high;

    CUTLASS_HOST_DEVICE TensorStorage() {}
  };

  struct Arguments {
    typename LowCollective::Arguments low;
    typename HighCollective::Arguments high;
    int low_k = 0;
    int high_k = 0;
  };

  struct Params {
    typename LowCollective::Params low;
    typename HighCollective::Params high;
    int low_k_tiles = 0;
    int high_k_tiles = 0;
    // The pipeline starts with the larger high-stream transaction size. Low
    // stages explicitly complete the unused byte delta.
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
  };

  template <class ProblemShape>
  static constexpr auto stream_shape(ProblemShape problem_shape, int stream_k) {
    if constexpr (CompactStreams) {
      cute::get<2>(problem_shape) = stream_k;
    }
    return problem_shape;
  }

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    auto low_shape = stream_shape(problem_shape, args.low_k);
    auto high_shape = stream_shape(problem_shape, args.high_k);
    return {
        LowCollective::to_underlying_arguments(low_shape, args.low, workspace),
        HighCollective::to_underlying_arguments(high_shape, args.high, workspace),
        args.low_k / KTile,
        args.high_k / KTile,
        TmaTransactionBytes};
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement(
      ProblemShape const& problem_shape, Arguments const& args) {
    auto problem_shape_mnkl = cute::append<4>(problem_shape, 1);
    int total_k = int(cute::get<2>(problem_shape_mnkl));
    bool partition_valid = args.low_k > 0 && args.high_k > 0 &&
                           args.low_k % KTile == 0 && args.high_k % KTile == 0 &&
                           args.low_k + args.high_k == total_k;
    auto low_shape = stream_shape(problem_shape, args.low_k);
    auto high_shape = stream_shape(problem_shape, args.high_k);
    return partition_valid &&
           LowCollective::can_implement(low_shape, args.low) &&
           HighCollective::can_implement(high_shape, args.high);
  }

  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const& params) {
    LowCollective::prefetch_tma_descriptors(params.low);
    HighCollective::prefetch_tma_descriptors(params.high);
  }

  template <class ProblemShape>
  CUTLASS_DEVICE auto load_init(
      ProblemShape const& problem_shape, Params const& params) const {
    auto low_shape = stream_shape(problem_shape, params.low_k_tiles * KTile);
    auto high_shape = stream_shape(problem_shape, params.high_k_tiles * KTile);
    auto low_inputs = LowCollective{}.load_init(low_shape, params.low);
    auto high_inputs = HighCollective{}.load_init(high_shape, params.high);
    // The kernel layer only inspects the first two tensors to derive M/N/L CTA
    // coordinates. The nested tuples remain private to this collective.
    return cute::make_tuple(
        cute::get<0>(low_inputs), cute::get<1>(low_inputs),
        low_inputs, high_inputs);
  }

 private:
  template <class Collective, class LoadInputs, class BlockCoord>
  CUTLASS_DEVICE static void load_stream(
      typename Collective::Params const& params,
      MainloopPipeline pipeline,
      PipelineState& pipe_state,
      LoadInputs const& load_inputs,
      BlockCoord const& blk_coord,
      int start_k_tile,
      int k_tile_count,
      uint32_t base_transaction_bytes,
      typename Collective::TensorStorage& shared_tensors) {
    using namespace cute;
    int lane_predicate = elect_one_sync();
    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                              typename Collective::SmemLayoutA{});
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                              typename Collective::SmemLayoutB{});
      Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()),
                                typename Collective::SmemLayoutSFA{});
      Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()),
                                typename Collective::SmemLayoutSFB{});

      auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;
      auto block_tma_a = params.tma_load_a.get_slice(0);
      auto block_tma_b = params.tma_load_b.get_slice(0);
      auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
      auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

      auto [m_coord, n_coord, unused_k_coord, l_coord] = blk_coord;
      auto broadcast_n = make_layout(
          make_shape(Int<size<1>(typename Collective::TileShapeSFB{}) /
                         size<1>(typename Collective::TileShape{})>{},
                     Int<cute::numeric_limits<int>::max()>{}),
          make_stride(_0{}, size<1>(typename Collective::TileShapeSFB{}) /
                                size<1>(typename Collective::TileShape{})));

      Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);
      Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);
      Tensor gSFA = gSFA_mkl(_, _, m_coord, _, l_coord);
      Tensor gSFB = gSFB_nkl(_, _, broadcast_n(n_coord), _, l_coord);

      Tensor tAgA = block_tma_a.partition_S(gA);
      Tensor tAsA = block_tma_a.partition_D(sA);
      Tensor tBgB = block_tma_b.partition_S(gB);
      Tensor tBsB = block_tma_b.partition_D(sB);
      Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);
      Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);
      Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);
      Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);

      auto k_tile_iter = make_coord_iterator(
          idx2crd(start_k_tile, shape<3>(gA_mkl)), shape<3>(gA_mkl));
      CUTLASS_PRAGMA_NO_UNROLL
      for (; k_tile_count > 0; --k_tile_count) {
        pipeline.producer_acquire(pipe_state);
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(pipe_state);
        int write_stage = pipe_state.index();
        copy(params.tma_load_a.with(*tma_barrier),
             tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, write_stage));
        copy(params.tma_load_b.with(*tma_barrier),
             tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
        copy(params.tma_load_sfa.with(*tma_barrier),
             tAgSFA(_, _, _, *k_tile_iter), tAsSFA(_, _, _, write_stage));
        copy(params.tma_load_sfb.with(*tma_barrier),
             tBgSFB(_, _, _, *k_tile_iter), tBsSFB(_, _, _, write_stage));
        if (params.tma_transaction_bytes < base_transaction_bytes) {
          cutlass::arch::ClusterTransactionBarrier::complete_transaction(
              tma_barrier, cute::block_rank_in_cluster(),
              base_transaction_bytes - params.tma_transaction_bytes);
        }
        ++k_tile_iter;
        ++pipe_state;
      }
    }
  }

  template <class Collective, class FrgTensorC, class BlockCoord>
  CUTLASS_DEVICE static void mma_stream(
      MainloopPipeline pipeline,
      PipelineState pipe_state,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      typename Collective::TensorStorage& shared_tensors,
      typename Collective::Params const& params,
      BlockCoord const& blk_coord) {
    using namespace cute;
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                            typename Collective::SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                            typename Collective::SmemLayoutB{});
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()),
                              typename Collective::SmemLayoutSFA{});
    Tensor sSFB = [&]() {
      if constexpr (size<1>(typename Collective::TileShape{}) >= 128) {
        return make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()),
                           typename Collective::SmemLayoutSFB{});
      } else {
        Tensor temp = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()),
                                  typename Collective::SmemLayoutSFB{});
        auto n = get<1>(blk_coord);
        return temp(make_coord(_, n % (size<1>(typename Collective::TileShapeSFB{}) /
                                        size<1>(typename Collective::TileShape{}))), _, _);
      }
    }();

    typename Collective::TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));

    Collective collective;
    Tensor tCrSFA = collective.partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma);
    Tensor tCrSFB = collective.partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);

    auto smem_tiled_copy_A = make_tiled_copy_A(typename Collective::SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA = smem_thr_copy_A.partition_S(as_position_independent_swizzle_tensor(sA));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);

    auto smem_tiled_copy_B = make_tiled_copy_B(typename Collective::SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB = smem_thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_SFA = make_tiled_copy_impl(
        typename Collective::SmemCopyAtomSFA{}, collective.get_layoutSFA_TV(tiled_mma),
        make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor tCsSFA = smem_thr_copy_SFA.partition_S(as_position_independent_swizzle_tensor(sSFA));
    Tensor tCrSFA_copy_view = smem_thr_copy_SFA.retile_D(tCrSFA);

    auto smem_tiled_copy_SFB = make_tiled_copy_impl(
        typename Collective::SmemCopyAtomSFB{}, collective.get_layoutSFB_TV(tiled_mma),
        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tCsSFB = smem_thr_copy_SFB.partition_S(as_position_independent_swizzle_tensor(sSFB));
    Tensor tCrSFB_copy_view = smem_thr_copy_SFB.retile_D(tCrSFB);

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));

    auto K_BLOCK_MAX = size<2>(tCrA);
    int read_stage = pipe_state.index();
    auto tCsA_stage = tCsA(_, _, _, read_stage);
    auto tCsB_stage = tCsB(_, _, _, read_stage);
    auto tCsSFA_stage = tCsSFA(_, _, _, read_stage);
    auto tCsSFB_stage = tCsSFB(_, _, _, read_stage);

    auto copy_kblock = [&](auto k_block) {
      copy(smem_tiled_copy_A, tCsA_stage(_, _, k_block), tCrA_copy_view(_, _, k_block));
      copy(smem_tiled_copy_B, tCsB_stage(_, _, k_block), tCrB_copy_view(_, _, k_block));
      using MMAOp = typename Collective::TiledMma::MMA_Op;
      cutlass::gemm::collective::fp4_shift_A(MMAOp{}, tCrA_copy_view(_, _, k_block));
      cutlass::gemm::collective::fp4_shift_B(MMAOp{}, tCrB_copy_view(_, _, k_block));
      copy(tCsSFA_stage(_, _, k_block), tCrSFA_copy_view(_, _, k_block));
      copy(tCsSFB_stage(_, _, k_block), tCrSFB_copy_view(_, _, k_block));
    };

    auto gemm_kblock = [&](auto k_block) {
      cute::gemm(tiled_mma,
                 make_zip_tensor(tCrA(_, _, k_block), tCrSFA(_, _, k_block)),
                 make_zip_tensor(tCrB(_, _, k_block), tCrSFB(_, _, k_block)),
                 accum);
    };

    pipeline.consumer_wait(pipe_state);
    copy_kblock(_0{});
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 1; --k_tile_count) {
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
        if (k_block == K_BLOCK_MAX - 1) {
          cutlass::arch::NamedBarrier::sync(
              thr_size(tiled_mma),
              cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          pipeline.consumer_release(pipe_state);
          ++pipe_state;
          read_stage = pipe_state.index();
          tCsA_stage = tCsA(_, _, _, read_stage);
          tCsB_stage = tCsB(_, _, _, read_stage);
          tCsSFA_stage = tCsSFA(_, _, _, read_stage);
          tCsSFB_stage = tCsSFB(_, _, _, read_stage);
          pipeline.consumer_wait(pipe_state);
        }
        copy_kblock(k_block_next);
        gemm_kblock(k_block);
      });
    }

    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
      if (k_block == K_BLOCK_MAX - 1) {
        cutlass::arch::NamedBarrier::sync(
            thr_size(tiled_mma),
            cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        pipeline.consumer_release(pipe_state);
        ++pipe_state;
      }
      if (k_block_next > 0) {
        copy_kblock(k_block_next);
      }
      gemm_kblock(k_block);
    });
  }

 public:
  template <class LoadInputs, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(
      Params const& params,
      MainloopPipeline pipeline,
      PipelineState pipe_state,
      LoadInputs const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator,
      int,
      int,
      uint32_t,
      TensorStorage& shared_tensors) {
    int high_start = CompactStreams ? 0 : params.low_k_tiles;
    load_stream<HighCollective>(params.high, pipeline, pipe_state,
                                cute::get<3>(load_inputs), blk_coord,
                                high_start, params.high_k_tiles, params.tma_transaction_bytes,
                                shared_tensors.high);
    // No low-stage write may alias a still-live high-stage read in the overlaid
    // tensor storage. The phase boundary is infrequent (once per output tile).
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      pipeline.producer_tail(pipe_state);
    }
    __syncwarp();
    load_stream<LowCollective>(params.low, pipeline, pipe_state,
                               cute::get<2>(load_inputs), blk_coord,
                               0, params.low_k_tiles, params.tma_transaction_bytes,
                               shared_tensors.low);
    __syncwarp();
  }

  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline, PipelineState pipe_state) {
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      pipeline.producer_tail(pipe_state);
    }
  }

  template <class FrgTensorC, class BlockCoord>
  CUTLASS_DEVICE void mma(
      MainloopPipeline pipeline,
      PipelineState pipe_state,
      FrgTensorC& accum,
      int,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& params,
      BlockCoord const& blk_coord) {
    // Execute the stock high collective first: it owns the single accumulator
    // clear. The low OMMA stream then accumulates into exactly the same C
    // fragment and the kernel enters one epilogue. Reduction order is
    // immaterial, while this keeps the high path byte-for-byte aligned with
    // the validated stock CUTLASS implementation.
    HighCollective{}.mma(pipeline, pipe_state, accum, params.high_k_tiles,
                         thread_idx, shared_tensors.high, params.high, blk_coord);
    pipe_state.advance(params.high_k_tiles);
    mma_stream<LowCollective>(pipeline, pipe_state, accum, params.low_k_tiles,
                              thread_idx, shared_tensors.low, params.low, blk_coord);
  }

  CUTLASS_DEVICE void mma_tail(MainloopPipeline, PipelineState, int) {}
};

}  // namespace hygemm
