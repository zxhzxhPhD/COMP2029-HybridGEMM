# Cost-aware floating allocation and offline FP4 quantization

This stage extends `floating_selection_fixed_counts.md`. Device kernels, the
MixLLM implementation, FP8 activation packing and high-partition FP8 weight
packing remain unchanged. The optional `hygemm_offline` loader changes only
output-channel selection and, for Q3, the prepacked low-partition weights.

## Budget and controls

For projection j, promoting one output channel costs `4*K_j` payload bits.
Each model retains its own Q1 payload cap, with less than 0.001 bits/weight
unspent and strictly less than 5.1 bits/weight. Audit actual `w4`/`w8` storage,
native padded scale buffers, and channel indices separately. The user's newer
payload requirement supersedes the initial exact physical-byte cap: metadata
may change, but is measured and disclosed. No persistent FP16 shadow or
additional runtime tensor is permitted.

Q2 reuses the Q1 score vectors and calibration windows. It ranks by score/K,
selects a cost-feasible prefix, then fills the residual with affordable rows.
This is a deterministic greedy baseline, not an exact knapsack solver. Near-full
budget matching can select negative benefits; report their count and weight.
There is no per-layer quota. Compare all four floating score families, original
integer scores, three random seeds and a uniform-per-projection quota control.
Both loss proxies receive the same three-seed check as Q1. Record every Q1 PPL,
Q2 PPL and their paired difference, including reversals and negative results.

All quality runs inherit the original kernel IDs/token buckets/concurrency per
fused layer even when H changes; audit actual dispatch rather than relying on
missing-shape fallbacks. Use full WT2 **validation**, unchanged FULL Graph with
Inductor disabled. The opt-in loader must reproduce Q1 block NLL before running
new allocations. Test data are not scored for selection.

## Q3 factorial comparison

Generate offline E2M1/E8M0 weight banks, preserving K32 groups and scale width.
For each group compare native RTN with exponents shifted by 0, -1 and -2.
Choose the candidate minimizing either unweighted squared weight error or
error weighted by `E[Q_A8(X)^2]` per input feature, from the same FP16 teacher
and 32 training windows. This is a diagonal approximation, not full output
reconstruction or GPTQ. Equal objective retains the native candidate.

The six fixed masks are Q1/Q2 integer, Q1/Q2 absolute loss and Q1/Q2 signed loss.
Test both quantizers with each mask. If a different Q2 family wins, add its Q2
mask and existing Q1 counterpart, without altering the six-mask quantizer
selection rule. Select the quantizer by mean validation NLL over those six
masks, then recompute both loss proxies and allocate again; also recompute a
different winning floating family if applicable. Separate fixed-mask quantizer
gain, mask-only gain, and joint reselection gain.

After the initial matrix favored original quotas, a documented bounded
follow-up reuses the new 32-window scores to reselect row identities at the
original per-projection counts. It evaluates both loss proxies on all four
models, with paired same-backward RTN controls. This separates score refresh
from quota movement without adding quantizer parameters. These registered
32-window candidates participate in the primary validation choice; paired
controls, extra seeds and 128-window sensitivity points do not.

The native RTN bank is a control: verify loaded bytes against the existing
native packer and compare full-model validation. Packed full-row banks are
offline calibration artifacts. Deployment gathers only selected low rows,
reuses native FP8 high rows and native scatter IDs, and retains the same six
runtime tensors. Persist file hashes and bank provenance.

Bounded calibration-size sensitivity uses 3B/7B, 32 versus 128 windows of 512
tokens, seed2029, both loss proxies, RTN and the chosen Q3 quantizer. The first
32 windows must match Q1. Q3 quantizer moments remain fixed to isolate score
sample count. Main comparisons use the predeclared 32-window seed2029 setting;
never choose the best seed. Larger-context calibration, GPTQ and budget curves
are separate follow-up experiments.

## Validation and handoff

`check_components.py` validates native byte parity, E2M1 ties, scale layout,
tails, empty partitions, complete GEMM outputs and changed-input Graph replay.
Run compute-sanitizer memcheck. Compare the unmodified Q1 scorer and the port
on 0.5B using deterministic teacher backward; require exact score-vector parity.
The historical nondeterministic teacher backward has measurable repeat variation
in loss scores, despite identical weight/output scores and forward losses.
Preserve the failed historical strict gate and repeated controls; keep historical
Q1/Q2 vectors frozen. Q3 computes paired RTN scores from the same X and dL/dY as
each bank score and measures their masks as controls, excluded from primary
selection. Determinism affects the offline teacher only, not serving kernels.
Every vLLM run audits shape, payload, physical storage, layer plans and batching
control. Compare block NLL with paired block bootstrap intervals; treat these
as descriptive development-set uncertainty, without multiplicity correction.

For the quality-stage 7B timing check, compare original, best Q2 and final
quality candidate. New partitions receive bounded tuning with existing kernel
variants; keep quality scoring on inherited plans. Measure actual FULL decode
at B1/8/64/128, prompts128/1024, one warmup plus five samples, CV<=5%, at most
three groups. Require whole-model confirmation before adopting microbenchmark
plans. This is a preliminary handoff, not the stage2 performance search.

Sources live in `integrations/vllm/allocation/`. Store outputs under the
three-stage `tmp` root, not Git. Save masks, exact H/shapes, payload and physical
bytes, quantizer/bank hashes, quality results, initial plans, timing and negative
results. The project default mask remains unchanged until explicitly adopted.

## Running

Use the existing editable plugin and frozen native libraries in the vLLM
environment. Point `HYGEMM_LIBRARY`, `MIXLLM_LIBRARY` and `LD_LIBRARY_PATH` at the
audited binaries/CUDA runtime. Set `RUN_OUT`, `Q1_DATA` and `FORMAL_DATA` to the
new run, completed Q1 run and original formal run, respectively. From this
repository, run these serially on the SM120 host:

```bash
python integrations/vllm/allocation/prepare.py --out "$RUN_OUT" --q1 "$Q1_DATA" --formal "$FORMAL_DATA"
python integrations/vllm/allocation/check_components.py --out "$RUN_OUT/components.json"
compute-sanitizer --tool memcheck --error-exitcode 3 python integrations/vllm/allocation/check_components.py --sanitizer --out "$RUN_OUT/memcheck.json"
python integrations/vllm/allocation/run_q2.py --out "$RUN_OUT"
python integrations/vllm/allocation/run_q3.py --out "$RUN_OUT"
python integrations/vllm/allocation/run_finalize.py --out "$RUN_OUT"
python integrations/vllm/allocation/inspect_allocation.py --out "$RUN_OUT"
python integrations/vllm/allocation/analyze.py --out "$RUN_OUT"
python integrations/vllm/allocation/report_tables.py --out "$RUN_OUT"
python integrations/vllm/allocation/plot_results.py --out "$RUN_OUT"
```

`run_finalize.py` also checks extra existing mask seeds when the selected Q3
candidate uses a fixed Q1/Q2 loss-proxy mask; the bank remains fixed. This
follow-up does not select the best seed and is not a quantizer-moment seed sweep.
If a small model selects a joint-reselected mask, a bounded 128-window follow-up
checks that family and its paired RTN control while retaining the 32-window
primary. These follow-ups are recorded separately from the initial matrix.
Full FP4 banks remain in the experiment directory. `archive_results.py` archives
compact evidence and bank hash manifests; `archive_selected_banks.py` can
separately archive the primary banks. Do not confuse a full-row calibration bank
with the smaller partitioned runtime representation audited inside vLLM.
