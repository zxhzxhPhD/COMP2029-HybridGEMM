# Profiling the frozen two-kernel serving paths

`integrations/vllm/profiling/` diagnoses Qwen2.5-7B batches 1, 8 and 64 using the
formal experiment's model views, channel selection and exact tuning plans.
It does not rebuild or modify either native library. The opt-in worker extension
adds NVTX labels during Graph capture and host-side measurement hooks; ordinary
serving does not import this extension.

The configuration retains FULL Graph, disabled Inductor, capture sizes
1/2/4/8/16/32/64/128, max_num_seqs=128, max_num_batched_tokens=2048, a 10 GiB KV
budget and FlashAttention 2. Prompt length is 128, with the formal seed/token
slices and 129 generated tokens. No model calibration or retuning is performed.

## Measurements

1. Without Nsight, one warmup and five full decode generations per method/batch
   reproduce the formal CUDA event metric. CV must be at most 5%, with up to
   three complete attempts preserved.
2. Nsight Systems captures 32 full-batch pure-decode steps after skipping eight.
   `node:nvtx-precapture` retains kernel nodes and maps them to projection labels
   created before profiling starts. Boundary-only synchronization starts/stops
   collection; analysis discards two steps at each end.
3. A separate unprofiled event control brackets the existing Graph replay with
   two events **outside** the Graph. It measures the same 28 interior steps,
   their Graph duration, outside-Graph remainder and host replay-call duration.
   No timing nodes are inserted into the captured model Graph. Both whole-run
   and ROI sampling CV must pass 5%.
4. Optional tiny Graph probes use existing PyTorch sleep/fill kernels plus
   CUDA memset operations. They test simple node/topology hypotheses and do not
   change the serving execution or constitute optimization results.

`analyze_trace.py` resolves cloned CUDA Graph node IDs back to capture-time
NVTX ranges. All 112 fused projections and their expected precision branches
must be present in every captured step. GEMM duration is the union of both
branches' GPU intervals, including their fused scatter epilogues. Separate
transpose/copy operations inside a projection are output-layout overhead.
Activation quantization includes scale initialization. Attention/KV, other model
kernels and auxiliary GPU operations are recorded separately.

Within each graph-start-to-next-graph-start period, exclusive category activity,
cross-category overlap (if any) and unoccupied GPU intervals sum exactly to the
period. Unoccupied intervals inside the Graph are not labeled Python/CPU time.
The unprofiled boundary control establishes the outside-Graph remainder, which
includes LM head, sampling, preprocessing and scheduler/host gaps. Host CPU API
durations overlap GPU execution and must not be added to these GPU intervals.

Node tracing can perturb host dispatch and outside-Graph gaps. Attribution is
accepted only if the traced **Graph duration** agrees with the independent event
control within 2%; total traced decode latency is not substituted for the
unprofiled performance result. Static registers/shared/grid attributes describe
the selected implementation, not achieved occupancy, bandwidth or stall causes.
The current host has no Nsight Compute hardware-counter access.

## Reproduce

```bash
ROOT=/root/code/HybridGEMM
PY=/root/autodl-tmp/hygemm-vllm026/bin/python
FORMAL=/root/autodl-tmp/20261024_235427_formal_decode/7B
OUT=/root/autodl-tmp/your_decode_profile
mkdir -p "$OUT"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export HYGEMM_LIBRARY=/path/to/original/libhygemm_serving.so
export MIXLLM_LIBRARY=/path/to/original/planned/mixllm_sm120.so

"$PY" "$ROOT/integrations/vllm/profiling/run_profiles.py" \
  --formal "$FORMAL" --out "$OUT" --probes
for kind in mixllm hygemm; do
  for batch in 1 8 64; do
    "$PY" "$ROOT/integrations/vllm/profiling/analyze_trace.py" \
      --trace "$OUT/${kind}_b${batch}.sqlite" \
      --out "$OUT/attribution_${kind}_b${batch}.json"
  done
done
"$PY" "$ROOT/integrations/vllm/profiling/summarize.py" \
  --formal "$FORMAL" --out "$OUT" --plots
```

Raw `.nsys-rep` and SQLite traces, event controls, per-layer resource/latency
tables, figures and analysis belong in tmp. Keep the original library hashes,
model/plan provenance and a source snapshot with the results. Empirical reports
must distinguish measured stage costs from unproven hardware explanations.
