# vLLM 0.26.0 integration: original MixLLM and two-kernel HyGEMM

This integration establishes one serving stack for the two implementations.
It targets a single RTX 5090 (SM120), Qwen2.5-0.5B, FP16 model activations and
KV cache, and dense linear layers. It is a smoke-test milestone, not a decode
performance or model-quality result.

## Release choice

The common stack is vLLM **0.26.0**, PyTorch **2.11.0+cu130**, Transformers
**5.13.0**, FlashInfer **0.6.14**, and Python **3.12**. vLLM is installed from a
prebuilt release wheel. Only the small integration plugin is editable; the
HyGEMM library and MixLLM extension are built separately.

vLLM 0.9.0 is not intrinsically unable to invoke the HyGEMM kernel. The kernel
is an external CUDA operation; its important requirements are SM120, a capable
CUDA compiler, correct stream/capture behavior, and a matching framework ABI
when using the PyTorch C++ API. Keeping 0.9.0 would, however, retain older
framework dependencies and fewer low-precision integration options.

0.26.0 is a deliberate stable release pin rather than a moving main/nightly.
Its release notes include further NVFP4/MXFP4 work and FlashInfer integration.
The source contains MXFP8 linear backends and newer quantization registration
interfaces. These are useful future baseline entry points, but do not imply
that every MX format/backend uses native low-precision MMA on SM120. SM100-only
backends, W4A16 fallbacks, and MXFP8 versus ordinary FP8 must still be distinguished.

References:

- [vLLM v0.26.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.26.0)
- [v0.26.0 quantization registry](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/layers/quantization/__init__.py)
- [v0.26.0 MXFP8 FlashInfer backends](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py)
- [MixLLM upstream v0.9.0 patch](https://github.com/microsoft/MixLLM/tree/00263510f9760d3a1d958a388e0e7c25961a7d59/vllm_v0.9.0_patch)

The installed Torch reports CUDA **13.0**. Native builds use the existing CUDA
**13.0.48** toolkit. `constraints-cu130.txt` also caps otherwise unbounded
compiler dependencies at 13.0: resolving FlashInfer without these constraints
selected CUDA 13.4 compiler packages. CUDA library component versions such as
cuBLAS 13.1 are not the toolkit/runtime version; the frozen package list and
`torch.version.cuda` are retained with the experiment. No driver update is made.
The dependency set also contains the standalone `nvidia-cuda-nvdisasm` 13.4.92
disassembler. It does not execute GPU code or replace the compiler/runtime;
native builds and inspection use the existing `/usr/local/cuda` 13.0 tools.

## Out-of-tree port

`integrations/vllm/hygemm_vllm` registers `mixllm` and `hygemm` through
`vllm.general_plugins`. Parent and worker processes load the same plugin.
No installed vLLM Python source or native extension is patched.

The MixLLM adapter carries forward the original five-tensor checkpoint schema:
`qweight`, `scale`, `zero`, `indices`, `is_int8_channel`. It adds the current
`QuantizationConfig` initialization, validates the supported deployment scope,
and correctly adjusts partition-local indices after QKV/gate-up loading.
It retains the original flatten/restore shape handling, bias semantics, CUDA
INT8 activation quantizer, two GEMMs, and original CUDA/PyTorch transpose choice.
The original checkpoint loader is tested against the equivalent RTN preparation
after vLLM's standard FP16 weight loading.

Both methods also accept `checkpoint_format="fp16_rtn"` for this experiment.
They load the same original checkpoint, use an explicitly supplied saved channel
selection, and prepare weights once after loading. QKV and gate/up use the same
vLLM fusion and original projection offsets in both implementations. Embeddings,
LM head, attention, KV cache and norms retain the common vLLM implementation.
This mode does not perform calibration, GPTQ, clipping or a sensitivity search.

The integration currently rejects tensor parallelism above one, non-FP16
activations and K not divisible by 128. The MixLLM wrapper conservatively requires
O divisible by 128; its original vector epilogue is not a general arbitrary-O
implementation. Precision partition sizes may have tails. The HyGEMM native
library continues to support arbitrary positive O/T and partition tails.

## Preserving the MixLLM baseline

Upstream source revision is `00263510f9760d3a1d958a388e0e7c25961a7d59`, with its
own CUTLASS revision `4370102f9dacab813282e1d67722fceb0b90a019`. The source checkout
and the existing local MixLLM reproduction remain untouched. `build_mixllm.py`
builds an isolated copy and writes `compatibility.patch` and a source manifest.

The compatibility edits are limited to:

1. Expressing three existing FP16 conversions/divisions explicitly for modern
   PyTorch extension flags that disable implicit half operators.
2. Joining only streams actually launched for the current request. Upstream
   waits on both low/high events even for pure partitions; an unused event from
   a preceding graph capture can cause an invalid-argument error.

The GEMM candidates, MMA, software unpack, group dequantization, pipeline and
epilogue are unchanged. The wrapper supplies readable zero padding behind the
logical channel-ID views because the original epilogue speculatively reads IDs
in tail tiles before its store predicate. It preserves the logical channel
counts and selected precision.

Smoke runs seed the original dispatch with stage 5 / ID 1. This is a conservative
valid configuration, **not an optimized MixLLM baseline**. Upstream's full search
includes 11-stage configurations that exceed SM120's shared-memory allowance.
Before performance comparisons, benchmark valid candidates and replace the
smoke plan, without silently changing the GEMM implementation.

## HyGEMM runtime

`libhygemm_serving.so` exposes a framework-independent C ABI. PyTorch owns all
input/output allocations and passes its current CUDA stream. The wrapper uses
the existing two-kernel factory and direct scatter epilogues. A host-only
`update_operands` method rebinds pointers at an unchanged shape, reusing the
workspace; the device GEMM implementation is unchanged.

Each layer caches instances by token count. Unseen shapes must be warmed before
CUDA Graph capture. Each call allocates fresh PyTorch output/activation buffers,
so outputs do not alias a hidden reusable layer buffer. Concurrent low/high
launches use a fork/join dependency on the supplied stream. Weights and maps
stay alive with the layer; one instance must not be used concurrently by
independent host callers/streams. Devices are fixed at initialization.

The new online activation packer generates E4M3 values and CUTLASS-layout UE8M0
scales directly on the GPU. Weight packing uses the same operation once during
loading, with E2M1 or E4M3 data as appropriate. The smoke quantizer uses K32 blocks
and `scale = 2**ceil(log2(amax / max_finite))`, with an identity scale for zero
blocks. This is conservative RTN; it is not an accuracy-tuned quantization policy.
The timing of activation quantization and its scale-padding initialization will
be part of any later end-to-end measurements.

The opaque `torch.library.custom_op` has a fake implementation for vLLM's
compilation path. Both methods use the same wrapper boundary. Worker auditing
uses a named worker-extension RPC, without enabling pickle fallback for
arbitrary callable serialization. Python call counters include warmup/capture
calls and are not a count of replayed GPU graph nodes.

## Install and build

The validated environment is `/root/autodl-tmp/hygemm-vllm026`. Keep it separate
from the existing kernel-probe Python environment.

```bash
ROOT=/root/code/HybridGEMM
OUT=/root/autodl-tmp/your_vllm_experiment
VENV=/root/autodl-tmp/hygemm-vllm026
python3 -m venv "$VENV"
"$VENV/bin/pip" install uv
"$VENV/bin/uv" pip install --python "$VENV/bin/python" \
  -c "$ROOT/integrations/vllm/constraints-cu130.txt" vllm==0.26.0 transformers==5.13.0
"$VENV/bin/uv" pip install --python "$VENV/bin/python" --no-deps \
  -e "$ROOT/integrations/vllm"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
cmake -S "$ROOT" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUTLASS_DIR=/root/code/cutlass
cmake --build "$OUT/build" --target hygemm_serving -j3
"$VENV/bin/python" "$ROOT/integrations/vllm/build_mixllm.py" \
  --upstream /path/to/pinned/MixLLM --out "$OUT/mixllm_build"
```

The MixLLM build retains the whole original candidate set and can take several
minutes. This is distinct from rebuilding vLLM, which is avoided. A package
mirror may be supplied explicitly to uv; freeze the resolved dependencies after
installation and verify `pip check`.

## Validate and prepare the next round

```bash
"$VENV/bin/python" "$ROOT/integrations/vllm/run_validation.py" \
  --model /path/to/Qwen2.5-0.5B \
  --selection /path/to/fixed/selection.json \
  --hygemm-library "$OUT/build/libhygemm_serving.so" \
  --mixllm-library "$OUT/mixllm_build/extension/mixllm_sm120.so" \
  --out "$OUT" --sanitizers
```

The suite validates dequantized-reference math, new input pointers at unchanged
shape, graph replay with changed inputs, pure/mixed partitions and the original
MixLLM CUDA transpose path. It then runs two 16-token prompts twice through
FP16, MixLLM and HyGEMM, checks finite logprobs and deterministic repetition
within each mode, and audits all 96 fused projection modules (24 layers × 4).
Eager and FULL Graph runs are separate; results and failures are retained under
tmp. The MixLLM five-tensor checkpoint must match the equivalent load-time RTN
path. Experiment reports include source/binary hashes and the dependency lock.

The validated Graph configuration is `CompilationMode.NONE` with
`cudagraph_mode="FULL"`, exposed by `smoke.py --graph --no-compile`. It captures
graphs while retaining the eager arithmetic kernels. The suite uses this mode
by default for its Graph checks. In the initial smoke, default Inductor changed
the selected RMSNorm implementations and the two quantized paths had maximum
fixed-prompt logprob differences of about 0.4 versus eager. With compilation
disabled, all three paths matched their own eager generation tokens and all 11
checked prompt-position logprobs exactly. Default compiled runs are preserved
as diagnostics, not accepted for the next speed/quality comparison. This small
control establishes a usable common mode; it does not prove general numerical
equivalence or attribute the drift to one specific compiled operator.

Before the next speed/quality round, fix a shared execution mode, attention
backend, KV precision, model revision, selection/calibration protocol and
request mix. Tune the actual model shapes for both paths; neither the inherited
4096-wide microbenchmark plans nor the smoke dispatch proves optimality for
896/4864-wide Qwen projections. A 10% **channel** map is not necessarily a 10%
**weight-element** budget; the preparation manifest reports both. Report
activation packing and output-layout costs, fixed-history numerical checks and
quality metrics separately from free-generation examples.
