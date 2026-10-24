# Small-batch dual-kernel optimization

The small-batch path keeps output-channel partitioning, MX quantization, channel
IDs, FP32 accumulation, FP16 output, and direct scatter unchanged. Two additions
are independently selectable: lightweight GEMM candidates and fused initialization
of the padded activation scale buffer. Original GEMM IDs 0–7 remain available.
MixLLM and the model/framework configuration are outside this change.

## Lightweight candidates

`src/kernel/output_channel_light.cu` adds IDs 8–27 for both FP4×FP8 and FP8×FP8.
The axes are output channels M, tokens N, and input channels K. Candidates use
M=16/32/64, N=8/16, K=128/256, 1/2/4 warps, and 2/3/4/6 pipeline stages. The
factory at the end of the source is the exact inventory. `hg_info` reports compiled
register, shared-memory, and local-memory usage for each partition, offline.

All warps cooperate in a `cp.async` ring. This removes the dedicated producer
warpgroup, TMA descriptors, and shared-memory output scratch of the original
candidate family. Each warp owns output rows and uses native SM120 block-scaled
M16N8K32 MMA. The epilogue scatters directly through the existing channel IDs.

Global FP4 weights stay nibble-packed. Sixteen logical FP4 values use an aligned
16-byte shared container: eight data bytes and eight zero padding bytes. The
`ldmatrix ... b4x16_p64` instruction expands the nibbles, followed by the required
two-bit shift of each operand word. FP8 uses ordinary LDSM loads. An XOR swizzle
of 16-byte shared segments serves both types. This does not remove FP4 padding
or make FP4 operands half the size of FP8 operands in the mixed MMA instruction.

Four K32 scale bytes are copied together from the existing K128/128-row scale
layout. External packed tensors and model checkpoints require no conversion.
Output/token tails are predicated; K must be a positive multiple of 128. A
partial K256 tile is zero-filled. The ring uses a CTA barrier before consumption
and before reusing a slot. Its final iterations drain with `wait_group 0` when
fewer than the nominal number of outstanding groups remain. Operand rebinding
retains the shape and updates the pointers before launch/Graph capture.

## Small activation-scale initialization

For FP8 pack with at most 32 rows, `pack_mx<false,true>` initializes the padded
scale bytes inside the pack kernel. The first warp in each K128 block of row 0
owns that block's padding. For each 32-bit word, the associated row is
`(word % 4) * 32 + word / 4`. Only words belonging to padding rows are set to
`0x7f7f7f7f`; valid rows retain their original scale stores. Owners write
disjoint words, so no inter-CTA synchronization is required. This removes the
separate scale `cudaMemsetAsync` node while preserving every output and scale
byte, including padding. FP4 pack and FP8 pack above 32 rows use the original
memset-plus-pack sequence.

## Tuning and acceptance

The experiment scripts live in `integrations/vllm/optimization/`. Only token
buckets 1/2/4/8/16/32 are eligible for changes. Every MixLLM plan and every
HyGEMM plan for tokens >=64 is copied unchanged from the frozen formal run.

1. `check_light.py` validates every candidate against the dequantized reference,
   including empty partitions, irregular output/token counts, K tails, pointer
   rebinding, and changed-input Graph replay. Run memcheck, synccheck, and
   racecheck on the reduced stress set.
2. `check_pack.py` compares old/new packed values and the entire scale buffer,
   including padding, for both formats and boundary row counts. Its small FP8
   checks also replay a Graph with changed inputs.
3. `tune_small.py` screens all 28 candidates per partition, combines top-three
   shortlists with serial/concurrent execution, then refines actual O/K/high
   shapes against a fresh measurement of the original plan. Switching requires
   a 2% measured margin. The resident proxy uses repeated Graph calls; the cold
   proxy evicts weights with an untimed 256 MiB read-modify-write operation and
   warms only the activation before timing a single Graph replay.
4. `search_model.py` evaluates complete policies in the real model: original
   compute, resident-tuned compute, cold-tuned compute, and uniform deep-stage
   ID21 compute. All use the same new pack implementation. Per-token policy
   switching requires at least 1% over original compute in prompt-128 controls
   with the original FULL Graph configuration. It selects whole measured
   policies, rather than combining unmeasured per-layer winners.
5. `run_comparison.py` measures old HyGEMM, new HyGEMM, unchanged MixLLM, and FP16
   across all eight batch sizes and both prompt lengths. It also compares
   old/new PPL and actual small-batch decode tokens/chosen-token logprobs.
6. `analyze.py` checks selection, plan hashes, actual per-layer dispatch, FULL
   Graph execution, sampling stability, quality, and a 2% large-batch regression
   gate. It writes the final summary and comparison table.

Microbenchmark wins are proposals, not model-level evidence. Lightweight FP4
does not necessarily win at small batch; preserve original compute when the
whole-model gate rejects the proposal. The experiment report records which
partition actually uses a lightweight candidate. PPL exercises preserved
prefill plans, so the separate decode-quality test is essential.

## Reproduction

Use the pinned SM120/CUDA/vLLM environment from `docs/formal_decode_protocol.md`
and the existing formal 7B calibration, weights, and plans. Set `FORMAL` to that
run's `7B` directory, `OUT` to a fresh directory outside the repository, and
`OLD`/`MIX` to the frozen native libraries. Do not reuse a completed output
directory for a different build: resume markers do not invalidate themselves
when a binary or input changes.

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
PY=/root/autodl-tmp/hygemm-vllm026/bin/python
mkdir -p "$OUT"
cmake -S . -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_DIR=/root/code/cutlass -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
cmake --build "$OUT/build" --target hygemm_serving -j2
export HYGEMM_LIBRARY="$OUT/build/libhygemm_serving.so"
export MIXLLM_LIBRARY="$MIX"
$PY integrations/vllm/optimization/run_small.py --formal "$FORMAL" --out "$OUT" \
  --old-library "$OLD" --new-library "$HYGEMM_LIBRARY" --mix-library "$MIX"
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --error-exitcode 3 \
  $PY integrations/vllm/optimization/check_pack.py --old-library "$OLD" \
  --out "$OUT/pack_memcheck.json" > "$OUT/pack_memcheck.log" 2>&1
$PY integrations/vllm/optimization/analyze.py --formal "$FORMAL" --out "$OUT" --plots
```

Retain raw tuning proposals, rejected model results, accepted plans, all samples,
quality outputs, sanitizer logs, library hashes, source snapshot, and completion
commit in the experiment directory. Source and protocol belong in Git; measured
results, plots, and compiled libraries belong in `tmp`.
