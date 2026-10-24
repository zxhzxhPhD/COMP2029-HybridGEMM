# SM120 two-kernel output-channel GEMM

This implementation partitions output channels into MXFP4 and MXFP8 weight
submatrices. Both kernels read the same MXFP8 activation and write FP16 values
directly into the final token-major output through a fused scatter epilogue.
It does not modify MixLLM, implement an integer kernel, or implement a fused
single-kernel precision switch.

## Data contract

For `W[O,K]`, `X[T,K]` and output `Y[T,O]`, each precision partition supplies:

- A contiguous weight submatrix with rows gathered into partition order.
- Correspondingly gathered K32 UE8M0 weight scales in CUTLASS
  `Sm1xxBlockScaledConfig<32>::tile_atom_to_shape_SFA` layout.
- An `int32` array mapping each partition row to its original output channel.

Both partitions share the same row-major E4M3 `X[T,K]` values and K32 UE8M0
activation scales in `tile_atom_to_shape_SFB` layout. The scale layouts are
independent of the selected candidate's tile size. No expanded FP8 weight copy
is created for the FP4 partition.

`Y[token, channel_ids[row]]` receives the corresponding dot product. All channel
IDs must be in `[0,O)`, be unique across both partitions, and together cover the
desired output. The library assumes that the caller has validated this map;
it does not launch a map-validation kernel on every invocation. Arbitrary row
order is supported; sorted channel IDs within each partition usually provide
better store coalescing. The random-order correctness case exercises the
general contract.

Accumulation is FP32, output is FP16, and the operation is `Y = W X^T` in the
equivalent column-major `[O,T]` representation. There is no C/residual input,
bias, activation function, or accumulation into existing Y in this version.

`K` must be positive and divisible by 128. `O`, `T` and precision partition
sizes need not be multiples of a CTA tile. A partial final tile is predicated;
the logical high-precision channel count is not rounded up to 128. The scale
storage must include the padding required by the CUTLASS scale layout.

An empty partition is omitted: 0% and 100% high precision launch one GEMM.
No zero-sized GEMM is initialized or launched. Device allocations must satisfy
CUDA/TMA alignment (ordinary `cudaMalloc` allocations do); all inputs, output,
maps and kernel instances must outlive their outstanding GPU work.

## Implementation

`src/kernel/output_channel_gemm.h` exposes a non-template `DeviceProblem`,
`Kernel` and `make_kernel` interface. An instance initializes its TMA
descriptors and workspace once and can subsequently run on a stream without
reinitialization. It borrows the input/output pointers and owns its workspace.
The problem and pointers are immutable for that instance; construct a new
instance when they change.
Do not overlap launches of the same instance on multiple streams: its internal
workspace belongs to that instance. Separate requests that execute concurrently
need separate instances/workspaces and disjoint output buffers; their immutable
weights and activations may be shared when appropriate.

`output_channel_variant.cu` builds two weight-type instantiations per candidate,
using the stock pinned CUTLASS SM120 block-scaled mainloop. This retains one
FP32 accumulator per result through the entire K reduction. It does not
reintroduce an INT32 group accumulator or software group output scaling.

The custom `ScatterStore` visitor maps each accumulator element's logical
coordinate to the final output address and stores FP16 directly. Passing void
C and D disables the stock epilogue's regular TMA destination store and output
shared allocation. There is no intermediate global output tensor and no
post-GEMM scatter kernel. Stores are scalar in this first implementation,
because adjacent partition rows can map to non-adjacent destination channels.
This is a performance tradeoff, particularly for long token dimensions.

Macro-configured variants use distinct C++ namespaces. This prevents
configuration-dependent template/inline symbols from being merged across
translation units by the linker.

## Candidate family and tuning

Tile dimensions below are `[output channels, tokens, K]`.

| ID | Tile | Mainloop stages | Schedule |
|---|---|---|---|
| 0 | 128×16×128 | auto | cooperative |
| 1 | 128×32×128 | auto | cooperative |
| 2 | 128×64×128 | auto | cooperative |
| 3 | 128×128×128 | auto | cooperative |
| 4 | 128×16×256 | auto | cooperative |
| 5 | 128×32×256 | auto | cooperative |
| 6 | 128×64×128 | 2 | cooperative |
| 7 | 128×64×128 | auto | ping-pong |

The benchmark first times each non-empty partition with all eight candidates.
It retains the best three per partition, then times the candidate pairs under
both serial and concurrent stream scheduling. This is a bounded two-stage
search, not an exhaustive proof of the globally fastest configuration. The
selection is based on measured whole-request latency, not the sum of isolated
GEMM times.

Concurrent requests use a fork/join dependency for each pair. The next request
cannot overwrite Y before both kernels of the previous request complete.
CUDA Graph timing includes both GEMMs and their dependency structure. The
benchmark checks that a mixed request contains exactly two GPU kernel nodes,
or one at the pure-precision endpoints.

The saved plan consists of shape, exact high-channel count, map convention,
low/high candidate IDs and execution policy. `--low-id`, `--high-id`, and
`--policy=serial|concurrent` replay it without tuning. The benchmark still
initializes the candidate family at process startup; a serving integration can
construct only the two chosen instances through the library interface.

## Correctness and timing boundaries

The test data use signed, nonconstant representable values and nonconstant
power-of-two K32 scales. High-precision weights and FP8 activations include
values not representable in E2M1, so accidentally treating the high partition
as FP4 is detectable. The CPU reference uses the decoded values for each
original output channel, independent of its packed partition row.

Small tests fully compare all output elements for every variant and include
random scatter order, tail tiles, K residues relative to K256 tiles, and empty
or single-channel precision partitions. Outputs are initialized to NaN to
detect missing writes; guard regions detect out-of-bounds output writes. Large
cases scan all outputs for finiteness and sample exact reference values,
explicitly including both precision partitions. A compute-sanitizer memcheck
case exercises all variants with nontrivial scatter and residues.

Measurements use prepacked resident tensors and repeated CUDA Graph requests.
Final timing defaults to three groups of 100 requests after warmup. FP16 uses
cuBLASLt on the same decoded input matrices, with FP32 compute and FP32
split-K reduction storage. Allocation, weight gather, scale layout conversion,
quantization and activation pack are excluded. Scatter stores and scheduling
dependencies are included. These tests do not measure model accuracy or
end-to-end LLM inference.

## Build and run

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
OUT=/root/autodl-tmp/your_output_channel_experiment
cmake -S /root/code/HybridGEMM -B "$OUT/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUTLASS_DIR=/root/code/cutlass
cmake --build "$OUT/build" --target hygemm_output_channel_bench -j2
/root/miniconda3/bin/python /root/code/HybridGEMM/scripts/run_output_channel_bench.py \
  --binary "$OUT/build/hygemm_output_channel_bench" --out "$OUT" --phase all
```

The runner saves environment/source hashes, per-candidate timings, numerical
checks, sanitizer logs, final CSV and replayable plans under the explicit
output directory. Experimental results are kept outside the Git repository.

A single fixed-plan call, for example:

```bash
"$OUT/build/hygemm_output_channel_bench" --out=4096 --tokens=128 --k=4096 \
  --high=410 --mapping=interleaved --low-id=2 --high-id=0 --policy=concurrent
```

The IDs above illustrate the interface; use the actual saved tuning plan for
the target shape. The two-kernel implementation is the reference against which
a later single-kernel precision-switch design should be evaluated.
