# Stage-three final evaluation

The stage-three runner is `integrations/vllm/final_eval/`. Inputs and results live under the three-stage experiment root, in `03_final_evaluation/runs/20261026_105600_final_evaluation`; generated checkpoints, banks, logs and measurements are not Git assets.

## Methods and frozen implementation

The primary matrix contains Qwen2.5 base 0.5B, 1.5B, 3B, 7B and Mistral-7B-v0.3. Each model has five methods:

| Method | Selection | FP4 bank | Runtime |
|---|---|---|---|
| FP16 | none | none | native vLLM FP16 |
| Mix-I | original integer score | integer RTN | original Mix device binary |
| Hy-I | original integer score | native floating RTN | stage-two K1 |
| Hy-F | frozen floating selection | native floating RTN | stage-two K1 |
| Hy-F-Q | same selection as Hy-F | input-moment-weighted clipping | stage-two K1 |

Mix-I is a controlled RTN deployment baseline, not a reproduction of the paper's best offline GPTQ/clipping quality. Its binary SHA-256 must remain `1d6190c2bb2effb24d1c12a5438ce051dc0967ec8cdf71e3120e207a38d426da`.

The accepted stage-two serving and producer binaries remain unchanged. Mistral uses an explicit opt-in Python MLP adapter to the existing exact SiLU producer. The additional CUDA change is confined to the independent format probe's reference online activation pack; it is not linked into the serving library.

For 3B/7B, Hy-I-Q is an additional selector-by-quantizer ablation: it retains the I mask and accepted I compute plan while using exactly the Hy-F-Q clipping bank. Together with Hy-I, Hy-F and Hy-F-Q this separates selection, clipping and their interaction. It receives six timing points, both PPL corpora, full tasks and cached decode quality; it does not replace the previously selected primary method based on final scores.

All new channel counts receive an actual-shape native-MMA search, followed by development-only whole-model speed gates. A bucket is adopted only with at least 1% first-group geometric-mean gain, no point over 2% slower, and a positive repeated-group gain with the same regression limit. Rejected SIMT arithmetic is not reintroduced. Once frozen, no final score selects a method, tile, seed or bank.

## Main measurement boundary

Use the existing vLLM 0.26.0 wheel environment, Torch 2.11 + CUDA 13.0, TP1, FULL Graph, Inductor mode0, FP16 KV, 10 GiB KV, no prefix caching and FLASH_ATTN. Capture buckets are 1/2/4/8/16/32/64/128; main max sequences is 128. Compare batch 1/2/4/8/16/32/64/128 and prompt 128/1024 with 129 generated tokens, greedy and ignoring EOS.

The decode timer spans actual pure fixed-batch FULL decode steps, including online packing, output handling and engine gaps, excluding prefill/mixed/tail steps, loading and capture. Assert at least 32 actual FULL timed steps. Each group uses one warmup and five samples; accept the first group with CV at most 5%, with at most three groups. Preserve all groups. Run an independent complete timing group for Mix-I, Hy-I and Hy-F-Q in every model to interpret small differences.

Mistral's B128/prompt1024 workload cannot fit the common 10 GiB KV budget. Mark the same point infeasible for every method; do not substitute a preemption workload or enlarge only one method's KV allocation.

## Quality and correctness

- WT2 test: all complete nonoverlapping 2048-token blocks, no added BOS, 2047 scored targets each. Preserve discarded tails. Historical stages have already seen this test set, so it is not blind.
- C4: pinned English validation shard and document IDs; sample 2048 documents from the first 4096 documents of the first shard, then concatenate with two newlines and score the first 256 complete 2048-token blocks per tokenizer. This is a fixed subset, not an unbiased estimate of all C4. Calibration uses a separately pinned training sample with exact content hashes disjoint from holdout documents.
- PPL sums all target NLL before exponentiating. Score groups of eight and compare the first four blocks with group one at an unchanged 0.001 mean-NLL tolerance. Save block scores and paired 1000-draw bootstrap intervals, seed 3071.
- HellaSwag validation 10042 documents, ARC-Challenge test 1172 documents and PIQA validation 1838 documents; full splits, zero-shot, base completion, no chat template. Pin lm-eval 0.4.13 task sources, dataset revisions, actual requests and tokenizer. Use official task result processing on saved native vLLM likelihoods, not software-dequant model scores.
- Task scoring has its own one-sequence engine and FULL capture bucket `[1]`. Training-only controls found a batching delta above the unchanged 0.001 mean-logprob gate with multiple device sequences. API submissions may group 32 requests, but the worker asserts at most one scheduled device sequence. This scoring protocol is separate from main decode throughput.
- Cached decode quality uses common true continuations, not divergent generations. For 3B/7B, cover batches 1/8/32/64/128 and both prompt lengths; compare exactly 32 common FULL targets per request. Sample full-vocabulary distributions on prompt ordinal zero at target indices 97/105/113/121, reporting KL(FP16 || candidate), total variation and top-1 agreement separately from target NLL. Optimized/old engineering controls retain mean absolute target-logprob tolerance 0.002 and maximum 0.05. Preserve greedy-token equality separately.

The short free-generation diagnostic also uses the enqueue barrier and records top-five logprobs. An initial unbarriered 3B I-mask development run differed on one request at its fifth generated token despite zero paired teacher-forced differences. Controlled 3B/7B I-mask repeats produced identical tokens and all 128 common-history logprobs matched exactly. Preserve the original discrepancy; its specific low-level cause was not localized. This protocol repair happened before final scores and changes no serving arithmetic.

## Ablations and generalization

The I/F selection by K0/K1 implementation 2x2 uses RTN for both masks; clipping is the separate Hy-F to Hy-F-Q comparison. `refresh_old_controls.py` binds the accepted stage-two old baseline plans before development checks: 3B reuses its accepted retuned old configuration, while 7B retains the accepted inherited configuration. Both masks have the same per-layer counts in these models, so the exact shapes match. Budget curves in 3B/7B use a common clipping bank with alpha 0/.25/.5/.75/1, integer cost selection, fixed-layer-quota floating selection, uniform layer allocation and a random alpha-one diagnostic. Share the zero endpoint and the primary floating alpha-one point. Measure batch 1/8/64 at both prompts plus both PPL corpora. The all-MX-W8A8 endpoint is explicitly over the primary payload budget.

Calibration robustness in 3B/7B recomputes both scores and clipping statistics for seeds 2029/2030/2031, 32 versus 128 samples, 512 versus 2048 length, and C4 training distribution. Freeze the algorithm and per-layer high counts, reuse paired starts where possible, and report every setting rather than selecting the best seed.

For 7B, disable pack fusion, the SiLU producer, two-path concurrency and light high-precision configurations one at a time. Include B32 because that is where some accepted light configurations matter. Run longer prompts 2048/4096 at B1/8/32, nonbucket batches 3/12/24/48, and prefill/TTFT at B1/8 and prompts 128/1024/4096. Record actual padded Graph descriptors. Dedicated startup instrumentation records loading/capture and layout/quantization host elapsed time; CUDA event spans include host gaps and bank I/O, so they are not isolated GPU quantizer cost.

Interpretation clarification of the frozen mechanism labels: `A-no-pack` sets `HYGEMM_PACK_MODE=0` instead of 3, restoring the old pack selection rather than removing quantization or the historical padding-initialization fusion. Mode 3 selects vector packing only at T>=16; B1/8 are consequently unchanged controls for this mechanism. `A-serial` forces every plan to serial and can also alter its original order; `A-no-light-high` replaces only high IDs >=8 with ID0. These describe the already frozen operations, not additional post-freeze experiments.

Nonbucket batches also receive paired K0-F-Q versus Hy-F-Q cached-decode correctness checks, at both prompt lengths, before freeze on validation and after freeze on C4. The existing engineering tolerances stay unchanged. Long-context measurements establish workload performance coverage; they are not a long-context task-accuracy claim.

The independent format probe compares FP16, MX-W4A8, MX-W8A8 and MX-W4A4 at actual 7B projection shapes, repeatedly replaying fixed device buffers under Graph. Report kernel-only and reference online activation pack plus GEMM separately. The device reports 100663296 bytes of L2; weight payload sizes cross this capacity differently across formats. Repeated buffers do not guarantee L2 residency, and these GEMM times do not isolate intrinsic MMA throughput. The exact-representable test inputs establish arithmetic correctness, not model quality equivalence. W4A4 has lower activation precision and a different K64 packed operand path. Do not equate its speed with equal-quality W4A8 inference.

Optional ordinary FP8 at 0.5B/7B is selected only if native smoke passes; record the actual backend rather than assuming an MX or tensor-core activation format. The installed-source compatibility audit explains why dense `mxfp4` fallback, serialized ModelOpt formats, AWQ/GPTQ and Mix paper PTQ are not interchangeable ready-made controls.

## Freeze, execution and acceptance

Run development gates and official task-metric processing before `preflight.py`, then generate `protocol.json` and `final_jobs.json` with `campaign.py`. `freeze.py` records source snapshots, versions, model/tokenizer assets, data/requests, banks, masks, exact plans, binaries, SASS and environment evidence. No final model scoring starts before this manifest. All final processes verify the frozen inputs; full hashes are checked again at acceptance. Post-freeze analysis files (`report.py`, `accept.py`, `archive.py`) do not choose or change inference inputs and are explicitly outside the execution-source guard.

Run the declared jobs with `serial_jobs.py`, holding one exclusive GPU lock across the queue. Final acceptance requires every declared job, native numerical gates, exact task counts, actual FULL execution, timing stability, unchanged Mix binary and input hashes. Unsupported or capacity-infeasible points stay visible. A repair affecting execution or the protocol requires a new freeze version and rerunning affected measurements; never silently overwrite a frozen result.

`accept.py` returns a nonzero exit status if any strict gate fails, even when the complete matrix has executed. Its `execution_complete` field is separate from `passed`. In that case the launcher stops before automatic reporting; run `report.py` separately to inspect and deliver the completed measurements with failed gates clearly labeled. Reporting never converts a failed gate into a pass or adds retries beyond the frozen protocol.

Report speed per model and batch group, paired quality uncertainty, physical storage and payload separately, budget Pareto relationships, failures and limitations. Do not average raw PPL across models or claim equal-quality acceleration between methods with different quality.
