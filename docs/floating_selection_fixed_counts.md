# Fixed-count floating-point channel selection

This experiment implements the budget-audit, error-diagnosis and fixed-count
parts of the first quality stage. It preserves the native HyGEMM library,
MixLLM, quantizers, graph configuration, and existing per-model tiling plans.
It does not redistribute high-precision counts across layers.

`integrations/vllm/quality/` contains the offline experiment tools.
`hygemm_vllm/quality_worker.py` is an opt-in, read-only worker extension. It
enumerates actual persistent packed weight/scale/channel-index storages, checks
the scale-size API against the analytical row-padding layout, verifies complete
channel coverage and absence of the original FP16 projection parameter, and
counts the remaining registered model parameters/buffers separately. KV cache,
graph pools, allocator slack and kernel workspace are outside weight bit budget.

For each original unfused projection O×K with H high-precision channels, payload
cost is `4*(O+H)*K` bits. Physical scales are computed after QKV/gate-up fusion;
each nonempty partition rounds its row count up to a multiple of 128. New masks
must have exactly the original H in every unfused projection, and their actual
packed representation must match the original byte budget.

## Inputs and comparisons

Calibration uses the original WT2 train tokenization and 32 windows of length
512. Seed2029 tokens are checked against the previous formal calibration. WT2
validation, pinned to the same dataset revision, is the development metric.
Historical test tensors are read only to retrieve/compare the original calibration
windows; test tokens are not scored or used to select methods.

Four scores are compared with the original integer-derived mask and three
random same-count masks:

- Weight reconstruction improvement: `sum((Q4(W)-W)^2-(Q8(W)-W)^2)` per row.
- Activation-aware output improvement: with native-quantized Xhat,
  `E[(Xhat Q4(W)-XW)^2-(Xhat Q8(W)-XW)^2]` per output channel.
- Absolute loss proxy improvement: `E[abs(t4+0.5*t4^2)-abs(t8+0.5*t8^2)]`.
- Signed loss proxy improvement: `E[(t4+0.5*t4^2)-(t8+0.5*t8^2)]`.

Here `tb = sum_K(dL/dW * (Qb(W)-W))`, obtained from the unchanged FP16 model on
each calibration window. These are heuristic loss proxies, not exact Hessians.
They compare the recoverable difference between FP4 and FP8 rather than assuming
every large FP4 error can be recovered by promotion. Sorting is stable within
each original projection. Negative scores can still be selected when the fixed
H constraint requires it; no cross-layer transfer or budget reduction occurs.

Native pack supplies both weight and activation quantization; there is no new
quantizer implementation. Scores are streamed by 1024 output rows to bound GPU
memory. An additional weight-only output score is saved for diagnosis but is
not part of the predeclared candidate family. The two best nonrandom families
by average validation-NLL delta on 3B and 7B receive seed2030/2031 checks. The
initial protocol covered those two models; after observing primary-seed model
dependence, the same families were extended to all four models before additional
small-model seeds were evaluated. Record this in `robustness_extension.json`.
All seeds are reported; the best seed is not silently selected.

## Error diagnosis and reference limits

Controlled software experiments use one HF SDPA backend and the same native
packed/dequantized values. Conditions are FP16, W16A8, mixed-W/A16, mixed-W/A8,
all-W4/A8 and all-W8/A8. The all-W8 and FP16 cases are diagnostic endpoints, not
budget-feasible final candidates. Sixteen evenly spaced 512-token validation
windows are shared by every condition.

Both FP16-input dense GEMM and FP32 dense GEMM references are retained. Native
mixed GEMM is evaluated in the same HF model, with full-output per-layer checks
against the FP32 reference on the first diagnostic window. Local tolerance is
atol=0.002, rtol=0.002, matching existing component checks. Counterfactual NLL
differences are not assumed additive.

The initial two-window pilot required whole-model software/native mean-NLL
agreement within 0.001. It failed despite identical packed values and passing
local checks. Preserve that pilot and the protocol amendment; do not present it
as a passed equivalence check. The expanded diagnostic reports full-window
software/native NLL discrepancy explicitly. Effects near that discrepancy
remain unresolved. All selector decisions use the actual frozen vLLM/native
path, not these approximate software counterfactuals.

## Native evaluation

Validation scores all complete 2048-token blocks, without added BOS, using
positions1–2047. It saves per-block NLL and checks the first four blocks at
batch1 against group8. FP16 and unchanged MixLLM receive the same validation
reference run. The experiment does not compare validation PPL against historical
test PPL. Paired confidence intervals resample validation blocks, not individual
tokens; they describe this development split and are not final-test claims.

For 7B, the original mask and both shortlisted score families are additionally
measured at B1/8/64, prompt128, one warmup plus five samples, CV<=5%, up to three
groups. All original capture sizes and max_num_seqs=128 remain fixed. The worker
asserts actual FULL execution for every measured fixed-batch decode step.

## Running

Use an existing formal four-model run and the frozen small-batch library/plans.
Run `fetch_validation.py --out "$OUT/data"` on a CPU host with pyarrow and
huggingface_hub; it downloads only the pinned validation parquet. Copy the text
and source manifest to the remote output directory. This avoids adding packages
to the frozen serving environment. Then, from the repository:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export HYGEMM_LIBRARY=/path/to/frozen/libhygemm_serving.so
export MIXLLM_LIBRARY=/path/to/frozen/mixllm_sm120.so
PY=/root/autodl-tmp/hygemm-vllm026/bin/python
$PY integrations/vllm/quality/prepare.py --formal "$FORMAL" --out "$OUT" \
  --plans-7b "$PLANS_7B"
$PY integrations/vllm/quality/run.py --formal "$FORMAL" --out "$OUT" --through all
$PY integrations/vllm/quality/analyze.py --out "$OUT"
```

The runner keeps one GPU process active at a time. Completed results have
script/input/binary identities; incompatible resume attempts fail rather than
silently reusing data. Archive a stale result before explicitly rerunning it.
Keep all raw scores, selections, model configs, budgets, per-block NLL, samples,
failed diagnostic pilots, dependency hashes and final source snapshot in the
experiment directory outside Git. No model weights are duplicated there.
