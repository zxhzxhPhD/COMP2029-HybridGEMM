# Four-model decode and PPL comparison

`integrations/vllm/formal/protocol.json` freezes the experiment settings for
Qwen2.5 base 0.5B, 1.5B, 3B and 7B on one RTX 5090. The three methods are vLLM
FP16, original MixLLM INT4/INT8 kernels, and the two-kernel MXFP4/MXFP8 HyGEMM.
Single-kernel HyGEMM is outside this experiment.

The common serving stack remains vLLM 0.26.0, Torch 2.11.0+cu130, the CUDA 13.0
toolkit, FP16 activations/KV cache and FlashAttention 2. Inductor is disabled.
The engine requests FULL Graph; FlashAttention's uniform-batch restriction can
make vLLM label the effective mode FULL_DECODE_ONLY. Every **measured decode
step** is checked to call `run_fullgraph`, so no measured step silently falls
back to eager/piecewise execution. PPL uses teacher-forced prefill on the same
engine and arithmetic path, not autoregressive decode graph timing.

## Shared channel selection and quantization

Each model receives a newly computed, frozen selection using 32 WikiText-2
**train** sequences of length 512, seed 2029. The full MixLLM score is

```
t[channel] = sum_K(gradient * (dequantized_INT4_weight - FP16_weight))
score[channel] = E[abs(t + 0.5 * t*t)]
```

The largest 10% of output-channel scores globally are selected, with a stable
sort for ties. Both methods use exactly that model's selection, including the
same QKV/gate-up fusion and channel offsets. The calibration streams gradient
and quantization-delta work by 256 output rows, avoiding a full-model FP32 delta
or parameter-gradient allocation for 7B. It retains the cross-weight terms in
the channel score; it is not a diagonal-only Fisher approximation.

Weights use the existing RTN policies: MixLLM group128 integer scales/zeros;
HyGEMM group32 E8M0 scales and E2M1/E4M3 values. No GPTQ, clipping, FP4-specific
selection optimization or post-result quantizer adjustment is added. Embeddings,
LM head, norms, attention and KV retain the common FP16 framework behavior.

10% channels is not necessarily 10% weight elements. Both totals are saved in
each calibration manifest and must appear with the results. The shared map is
derived from integer quantization perturbations; this is a controlled map
comparison, not each format's best attainable model accuracy.

## Offline autotiling

The actual fused layer dimensions and exact high-channel counts are extracted
from the model and selection. Token buckets are
`1,2,4,8,16,32,64,128,256,512,1024,2048`, covering decode and the prefill/PPL cap.

To bound search cost while adapting to every actual partition width:

1. Group by total output width O, input width K and pure/mixed partition type.
   Select an actual layer with median high-channel count as representative.
2. Screen all explicit legal original MixLLM stage5/stage11 candidates on that shape.
   Reject candidates exceeding the queried 101,376-byte shared-memory limit or
   thread limit before launch. For HyGEMM, screen eight candidates independently
   per partition, then the top3×top3 pairs under serial/concurrent scheduling.
3. Refine the best three whole-layer plans on **every exact O/K/high count** and
   token bucket from the model. Save the selected plan and all raw samples.

Microbenchmarks use resident random inputs, the actual channel maps, online
activation packing and output-layout work, with three groups of 32 Graph calls.
They check finite sampled outputs and a decoded FP32 reference before timing.
This is a bounded tuning proxy, not an exhaustive global-optimality claim or a
replacement for whole-model measurements. Identical dimensions/counts share a
plan even when the layer's channel-ID distribution differs.
The original invalid-ID stage3 fallback is not included in this explicit
candidate inventory. The search therefore does not cover every instantiable
configuration in the original source.

At runtime the O/K/high key is exact. An unlisted token count uses the smallest
larger tuned token bucket's **configuration**, not padded logical GEMM data.
Counts above the largest bucket or missing partition shapes raise an error.
Instances are still initialized at the actual shape and warmed before capture.
The worker audit records which plans were applied.

### MixLLM host dispatcher

`build_mixllm.py --planned` exposes `config_info` and `gemm_planned` using the
unchanged original candidate templates and macros. The generated compatibility
patch adds only metadata queries and explicit host dispatch, in addition to the
previous half-operator and empty-stream-event fixes. Original MMA, unpack,
dequantization, pipeline and epilogue code remain unchanged.

The explicit dispatcher avoids the original coarse M/8 and high/1000 file-cache
keys, which cannot represent independent plans for all exact partition widths.
It also avoids filesystem reads/writes in formal forward calls. The original
smoke entry point remains available when no plan is supplied. This is not an
SM120 integer-kernel redesign.

## Decode measurements

The matrix has batches `1,2,4,8,16,32,64,128` and initial prompt lengths 128/1024.
Prompts are fixed token slices, distinct within a batch and identical across
methods for that model. Prefix caching and detokenization are disabled. Greedy
generation ignores EOS and produces exactly 129 tokens per request: the first
from prefill, followed by 128 decode steps. The maximum batched-token budget is
2048 and the KV allocation is fixed at 10 GiB for all methods.

Chunked prefill may admit requests at different steps. The timing worker reads
the scheduler's actual computed-token positions and retains only steps in which
all scheduled requests are already past their prompts, each schedules one token,
and the active batch equals the requested batch. Prefill, mixed prefill/decode
and shrinking-batch tail steps are excluded. At least 32 full-batch decode steps
must remain. Per-trial measured step counts and context ranges are saved; they
are not assumed to equal 128 for every batched workload.

CUDA events bracket `execute_model` through `sample_tokens`, including LM head
and sampling. There is no synchronization per step. After generation, the span
from the first retained start event to the last retained end event is divided
by the retained step count. The span retains inter-step GPU idle gaps caused by
the engine/host, rather than summing only kernel durations. Final endpoint CPU
postprocessing is outside this metric. Full-request wall time is also retained,
but includes prefill and is not labeled decode latency.

The worker separately counts actual FULL Graph dispatches and asserts that this
count equals the measured decode-step count. Event objects are preallocated.
Reported tokens/s is `batch * 1000 / median_ms_per_step` for the measured fixed-
batch interval, not API request/s or prefill-inclusive throughput.

Each cell has one warmup and five measured generations. The population standard
deviation divided by the mean of the five per-trial decode times must be ≤5%.
A failed cell receives another warmup and a fresh five-sample group, up to three
attempts total. All attempts are preserved. Use the first passing group; if none
passes, mark the cell unstable rather than silently filtering individual samples.
Comparison aggregates exclude pairs whose relevant stability gate failed.

There is one GPU experiment process at a time. No frequency locking, cooldown,
thermal admission, lengthy noise investigation or additional environment gates
are used. Method order rotates by model. Models run sequentially to fit 7B and
the common KV budget on one 32 GB GPU.

## PPL

Tokenize the full WikiText-2 test text without adding BOS, split into complete
nonoverlapping 2048-token blocks, and score positions 1–2047 of each block using
vLLM prompt logprobs. The unused final partial block is explicitly counted in
the output. Sum target-token negative logprobs in CPU double precision and
report `exp(total_NLL / scored_targets)`. This is test PPL, not training or
calibration loss.

Evaluate all complete blocks with request group size eight. Repeat the first
eight blocks with group size one and require maximum per-block mean-NLL
difference ≤0.001. All methods must have identical token hashes, block counts
and scored-target counts within each model. Save each block's NLL. A single PPL
value is reported per model/method; batch invariance is a separate control,
rather than repeating the full corpus at every decode batch size.

Lower PPL is better. A faster path with a higher PPL is not an equal-quality
speedup. These results do not establish performance on other tasks/datasets or
the best quality achievable after format-specific calibration.

## Reproduce

Results/builds/models stay outside the Git repository. Reuse the isolated vLLM
environment and native HyGEMM library from the integration milestone.

```bash
ROOT=/root/code/HybridGEMM
OUT=/root/autodl-tmp/your_formal_decode_run
PY=/root/autodl-tmp/hygemm-vllm026/bin/python
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export HYGEMM_LIBRARY=/path/to/libhygemm_serving.so
export MIXLLM_LIBRARY="$OUT/mixllm_build/extension/mixllm_sm120.so"

"$PY" "$ROOT/integrations/vllm/build_mixllm.py" --planned \
  --upstream /path/to/pinned/MixLLM --out "$OUT/mixllm_build"
# Put the fixed WT2 text files in $OUT/data and pinned model snapshots under
# /root/autodl-tmp/models/Qwen2.5-{0.5B,1.5B,3B,7B}.
"$PY" "$ROOT/integrations/vllm/formal/run_matrix.py" \
  --out "$OUT" --models /root/autodl-tmp/models
"$PY" "$ROOT/integrations/vllm/formal/analyze.py" --out "$OUT" --plots
```

The matrix runner resumes completed calibration/tuning and completed decode
cells. A resumed measurement must match its saved token/batch protocol and plan
hash. `pipeline_status.json` records the current phase or failure. Final
`decode.csv`, `quality.csv`, plots, raw result JSON, calibration manifests and
tuning inventories support the report; setup time is never substituted for
decode time.

Run `formal/verify.py --out "$OUT"` after the matrix completes. It checks the
complete cell grid, raw-sample statistics and Graph dispatch counts, shared
channel selections, exact runtime plan choices, and PPL token identity. Output
hash variation across greedy repetitions is recorded separately from the timing
stability gate; PPL always scores fixed supplied token histories.
