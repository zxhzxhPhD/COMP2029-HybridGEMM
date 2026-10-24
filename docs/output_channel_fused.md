# SM120 single-kernel output-channel precision switching

The fused implementation executes MXFP4 weights × MXFP8 activations and
MXFP8 weights × MXFP8 activations in one kernel. Precision is uniform within
an output tile and can change when a persistent CTA receives another output
tile. Both paths use the same tile, stage count, pipeline ring and schedule.
The two-kernel implementation remains available as the performance reference.
MixLLM and the pinned upstream CUTLASS source are unchanged.

## Data and API contract

`src/kernel/output_channel_fused.h` adds `make_fused_kernel(id, low, high,
stream, audit)`. The two `DeviceProblem` objects use the prepacked data and
channel-map contract described in [output_channel_gemm.md](output_channel_gemm.md):

- Separate compact FP4 and FP8 weight partitions and their K32 UE8M0 scales.
- One shared FP8 activation tensor and activation-scale tensor.
- Two row-to-original-channel maps; all IDs are valid, unique across both
  partitions, and cover the output channels.
- One FP16 output, with FP32 accumulation and direct scatter stores.

The factory requires identical activation/output pointers and O/T/K dimensions,
and `low.rows + high.rows == O`. K is a positive multiple of 128. O, T and each
partition's row count may have tails. Either partition may be empty. An unused
partition receives a valid dummy TMA descriptor but is never scheduled or read.
No extra global output, scatter kernel, activation conversion or expanded FP8
weight copy is introduced.

Instances own their workspace and borrow tensors/maps. Keep all pointers alive
until GPU work completes. Do not execute one instance concurrently on multiple
streams; independent requests need independent instances/workspaces and outputs.
Maps are caller-validated, as in the two-kernel API.

## Virtual output tiles and switching

For CTA_M = 128, the scheduler sees:

```
low_tiles  = ceil(low.rows / 128)
high_tiles = ceil(high.rows / 128)
virtual_M  = (low_tiles + high_tiles) * 128
```

An output tile with M index below `low_tiles` uses the low partition; subsequent
tiles use the high partition with local M index `m - low_tiles`. Each partition
keeps its actual logical shape in its TMA descriptors. Padding affects scheduling
and predication only; it does not promote additional channels to FP8.

The custom collective wraps the two stock CUTLASS mainloops, retains the stock
SM120 kernel/CLC scheduler, and selects the load and MMA path once per output
tile. The epilogue selects its row map once per CTA and directly scatters valid
elements to `Y[token, channel_ids[row]]`. Each stock MMA call starts a fresh FP32
accumulator for its own output tile. Precision never changes within a K reduction.

Producer and consumer independently derive precision from their own work tile.
There is no shared mutable precision flag: producer and consumer can be handling
different output tiles at the same time. Scheduler order is not a guarantee that
all low work finishes before all high work begins.

## One shared ring, two exact TMA transaction counts

Equal shared allocation does **not** imply equal TMA expected bytes. The FP4
mixed path occupies a byte-container shared layout, but its weight transfer
transaction accounts for FP4 data. Activation and scale transfers must also be
included. The implementation takes the exact constants from each stock
collective's `TmaTransactionBytes`, rather than using `sizeof(TensorStorage)`.

For example, tile 128×16×128 has:

| Path | Expected bytes per K stage | Kernel shared storage, four stages |
|---|---:|---:|
| MXFP4 × MXFP8 | 11,264 | 83,968 |
| MXFP8 × MXFP8 | 19,456 | 83,968 |

`MainloopPipeline` initializes the barriers once. At an output-tile boundary,
the producer creates a stock pipeline **view** over the same barrier arrays,
with the selected byte count and `InitBarriers=false`. Its ordinary
`producer_acquire` issues the exact `arrive_and_expect_tx` before the four TMA
copies. This is not a second pipeline or a barrier reset. The caller's stage
index and phase keep advancing across output tiles. Consumer wait/release uses
the same barriers and does not need a transaction-count change.

There is no drain/refill on a precision switch; live stages can belong to
different precisions. The implementation statically checks equal dispatch
policies, tile/SMEM layouts, pipeline types and accumulator fragments. The shared
tensor union also checks standard layout, all member types/offsets, total size
and alignment. Thus both typed views overlay one allocation with the same common
initial sequence. This implementation is restricted to the current cluster
shape 1×1×1 and pinned CUTLASS revision; a different layout needs revalidation.

The initial prototype added the high-path byte-count delta with an extra
`expect_transaction` per stage. The final version uses the exact-count view to
remove that extra instruction. It also hoists scatter partition selection out
of the per-element store loop.

## Candidates, validation and comparison

The candidate IDs match the eight configurations in `output_channel_gemm.md`.
For each fused candidate, tile, stage count and schedule are common to both
precisions. They are never selected independently per partition. Auto stages
resolve to 4, 4, 3, 2, 2, 2, 2 and 3 for IDs 0 through 7 on the pinned build.

The benchmark verifies all candidates before timing. It shares the dual-path
test generator and independent decoded CPU reference. Small cases check every
element, including random scatter order, empty partitions, single-channel
partitions and K/tile residues. Larger cases scan all outputs for finite writes,
check output guards, and compare 512 reference coordinates per launch.

Optional `audit` points to six zero-initialized device `uint32_t` counters: low
tiles, high tiles, producer precision switches, consumer precision switches,
low K stages and high K stages. The benchmark checks exact tile/stage totals.
`--require-switch` requires **each tested variant** to observe both producer and
consumer switches within a CTA. Stress shapes have enough work to exercise
persistent switching at short K, a K256 residue and repeated stage-ring wraps.
Audit is disabled during timing. Memcheck/synccheck cover tail/scatter cases,
and additional cases exercise actual switching for cooperative and ping-pong
schedules.

The runner tunes eight fused candidates and replays the previously saved best
dual plan on the same device inputs, in the same process. Previously unmeasured
shapes obtain a dual plan from the frozen original dual executable. This retains
the original dual kernel code and its ability to use different tiles per path.
The final comparison alternates five groups of 100 requests after warmup, and
asserts exactly one GPU kernel node per fused request versus two for a mixed
dual request. This bounded search is not a global optimality proof.

Primary timing is CUDA Graph GPU elapsed time. Prepacking, allocation,
descriptor initialization and online quantization are excluded; scatter and
request dependencies are included. `--measure-eager` additionally measures
ordinary launches with the same chosen plans, reporting CPU submission time,
CUDA-event time and wall time per request over an asynchronous request batch.
Eager event time can include host-induced GPU idle gaps. Eager results describe
this C++ launch wrapper, including the dual path's stream fork/join event APIs;
they must not be presented as isolated kernel speedups or synchronous request
latency. The eager path is not separately autotuned.

Results, raw samples, environment/source hashes and replayable plans belong in
the explicit tmp output directory. The experimental report records performance
including regressions; fusion does not guarantee a speedup when the dual
baseline already uses concurrent graph launches.

## Reproduce

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
OUT=/root/autodl-tmp/your_output_channel_fused_experiment
cmake -S /root/code/HybridGEMM -B "$OUT/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUTLASS_DIR=/root/code/cutlass
cmake --build "$OUT/build" --target hygemm_output_channel_fused_bench -j3
/root/miniconda3/bin/python /root/code/HybridGEMM/scripts/run_output_channel_fused.py \
  --binary "$OUT/build/hygemm_output_channel_fused_bench" \
  --dual-binary /path/to/frozen/hygemm_output_channel_bench \
  --dual-plans /path/to/dual/tuning_plans.json --out "$OUT" --phase all
```

For a fixed plan, pass `--fused-id`, `--low-id`, `--high-id` and
`--policy=concurrent|serial` to `hygemm_output_channel_fused_bench`. For only
correctness, use `--check-only --audit`, optionally `--full-check` and
`--require-switch`. A fixed-ID run instantiates only that fused candidate.
