# Milestone 1.5: Execution Policy and Layout Contract

Status: accepted for the Milestone 2 implementation. This is a versioned v1
contract; measurements from the complete TMA mainloop may refine alignments and
stage counts, but must not silently change its semantic invariants.

## Decision

The calibration/scoring interface may continue to reason about individual K32
atoms. The default executable promotion mask, however, is K64-paired: adjacent
K32 atoms `(2j, 2j + 1)` receive the same precision. This preserves the native
K64 W4A4 OMMA for the dominant low-precision path.

An arbitrary-K32 mode is retained only as an explicit, bounded fallback. A
mixed pair (LH or HL) is executed as one generic K32 W4A4 QMMA plus one K32
W4A8 QMMA. The generic all-low K32 path is a benchmark/reference path and is
not a production baseline.

Milestone 2 implements the paired-only policy first. The singleton fallback is
part of the layout vocabulary so it can be added without replacing the address
model, but it is not enabled by default.

## Scope and invariants

- The v1 execution plan is static per layer and shared by every CTA_M and CTA_N
  tile. A CTA-dependent or request-dependent promotion mask is out of scope.
  This is what allows one offline weight pack and one online activation pack to
  serve the whole GEMM without weight duplication or mainloop gathers.
- Logical scale granularity remains K32 for both operands. A K64 OMMA record
  therefore owns two adjacent UE8M0 scale factors (the instruction's 2X scale
  vector); each K32 QMMA record owns one.
- Every original weight value is stored exactly once as E2M1. Every activation
  value is stored exactly once, either as E2M1 or E4M3. There are no complete
  dense A4 and A8 copies.
- Reordering is a lossless partition of the K reduction. All class loops write
  the same FP32 accumulator, and the kernel performs one epilogue.
- Original K order is represented in the execution plan, not followed by the
  MMA mainloop. The sum over K permits class-order execution.

The layer-global-mask restriction is deliberate. If precision depended jointly
on CTA_M and CTA_N, neither a single packed weight tensor nor a single packed
activation tensor could satisfy all CTAs: the implementation would need weight
duplication, gather/tile-ID addressing, or an additional factorization rule.

## Pair classification and execution

Adjacent original K32 atoms form one logical pair:

| Class | Physical activation data | MMA sequence for one pair |
| --- | --- | --- |
| LL | one compact A4 K64 record | one K64 E2M1 x E2M1 OMMA |
| HH | two adjacent compact A8 K32 records | two K32 E2M1 x E4M3 QMMAs |
| MIX | paired A4-low and A8-high K32 records | alternating low QMMA and high QMMA |

LH and HL have the same packed MIX representation. The plan records the
original atom ID for each half, while physical slot 0 is always the low half
and slot 1 is always the high half. This removes a branch from the mainloop.

MIX is physically represented by two type-homogeneous compact substreams, but
the streams share a one-to-one record index and are consumed together. They
must not become two independent, fully grouped K loops: the instruction
microbenchmark found alternating low/high issue materially faster than first
issuing every low QMMA and then every high QMMA.

## Physical streams

The packer produces the following logical streams. Value and scale storage are
separate, matching the block-scaled MMA/TMA interfaces.

```text
LL:
  W4_LL64, SW_LL[2 scales/record]
  X4_LL64, SX_LL[2 scales/record]

HH:
  W4_HH32[2 records/pair], SW_HH[1 scale/record]
  X8_HH32[2 records/pair], SX_HH[1 scale/record]

MIX (optional fallback):
  W4_MIX_LOW32,  SW_MIX_LOW     X4_MIX_LOW32,  SX_MIX_LOW
  W4_MIX_HIGH32, SW_MIX_HIGH    X8_MIX_HIGH32, SX_MIX_HIGH
```

The weight streams are generated offline from the static layer plan. Activation
streams are generated online for each forward pass by a fused producer when
possible (GEMM epilogue or norm output), with a standalone pack kernel retained
as the measurable fallback.

The storage above is a semantic contract, not a promise of flat row-major scale
bytes. The implementation must use the CUTLASS SM1xx block-scale factor layout
and the value swizzle/alignment required by its chosen TMA collective. The plan
exposes record counts and aligned byte offsets; the collective owns the exact
TMA tensor maps and shared-memory layouts.

Each class/substream is padded independently to the selected CTA/TMA tile
alignment. Padding values are zero and padding scales are unity. If the original
K contains an odd number of K32 atoms, a zero K32 partner is introduced before
classification.

## Addressing contract

The MMA mainloop uses contiguous, affine TMA traversal inside each stream:

1. iterate the LL stream with K64 OMMA;
2. iterate the HH stream as pairs of K32 W4A8 QMMA;
3. optionally iterate matching MIX-low/MIX-high records and alternate the two
   K32 QMMA types;
4. execute one epilogue from the shared FP32 accumulator.

Values of different element types use different TMA descriptors. The mainloop
does not read a per-atom mask, chase a tile-ID list, or perform a prefix sum.
Original atom-ID maps are pack-time metadata and are not in the steady-state
MMA loop.

The compact tensors are tile-major at their producer/consumer boundary: weights
are packed per CTA_M-compatible tile and activations per CTA_N-compatible tile,
while every tile uses the same K-pair class order. Exact CTA shapes are a
collective tuning parameter rather than part of the serialized plan version.

## Plan/packer/collective interface

The concrete C++ types may change, but the boundary carries this information:

```cpp
enum class ExecutionPolicy { K64Paired, K32SingletonFallback };

struct StreamSpan {
  void* values;
  void* scales;
  int logical_records;
  int padded_records;
  int64_t value_stride_bytes;
  int64_t scale_stride_bytes;
};

struct ExecutionPlanV1 {
  uint32_t version;
  ExecutionPolicy policy;
  int original_k;
  int padded_k;
  int ll_pairs;
  int hh_pairs;
  int mixed_pairs;

  // Pack-time maps from compact records to original K32 atom IDs.
  Span<int32_t> ll_pair_atom0;
  Span<int32_t> hh_pair_atom0;
  Span<int32_t> mixed_low_atom;
  Span<int32_t> mixed_high_atom;

  // Per-stream aligned offsets, strides, and padding counts.
  StreamLayout ll;
  StreamLayout hh;
  StreamLayout mixed_low;
  StreamLayout mixed_high;
};
```

The packer consumes the atom-ID maps. The collective receives only compact
stream spans/counts plus its TMA descriptors. This separation prevents
calibration metadata from leaking into the hot mainloop.

## Rejected v1 address models

- Full dense A4 plus full dense A8 tensors: duplicates activation traffic and
  storage, defeating the purpose of sparse promotion.
- A dense A4 base plus compact A8 override: still reads or stores promoted A4
  values unless an additional skip/gather mechanism is introduced.
- Per-CTA tile-ID lists in the mainloop: prevent a simple affine TMA walk and
  make producer packing and weight reuse harder.
- Generic K32 for every low atom: discards the architecture's K64 OMMA advantage.
- Two fully separated MIX loops: valid numerically, but slower in the isolated
  instruction benchmark than paired alternating issue.

## Gate for revisiting the policy

The paired-only decision should be revisited only with end-to-end evidence that
the additional accuracy of arbitrary K32 masks outweighs the measured MIX
instruction cost and its extra pack/TMA machinery. That comparison must include
pack plus GEMM time; snapshot Frobenius error alone is not sufficient.
