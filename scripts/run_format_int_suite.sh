#!/usr/bin/env bash
# Re-run only the original seven shapes, adding INT8 and supported INT4 PTX.
set -euo pipefail
OUT=${1:?Usage: run_format_int_suite.sh ABSOLUTE_NEW_OUTPUT_DIRECTORY}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
CUTLASS_DIR=${CUTLASS_DIR:-/root/code/cutlass}
mkdir -p "$OUT/bin" "$OUT/source"
if [[ -e "$OUT/results.jsonl" || -e "$OUT/mma.jsonl" ]]; then
  echo "Refusing to overwrite existing measurements in $OUT" >&2
  exit 1
fi
cmake -S "$ROOT" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUTLASS_DIR="$CUTLASS_DIR" > "$OUT/configure.log" 2>&1
cmake --build "$OUT/build" --target hygemm_sm120_format_probe hygemm_sm120_format_probe_tuned \
  hygemm_sm120_format_int_probe hygemm_sm120_format_mma_probe -j2 > "$OUT/build_final.log" 2>&1
cp "$OUT/build/"hygemm_sm120_format*probe "$OUT/bin/"
cp "$OUT/build/hygemm_sm120_format_probe_tuned" "$OUT/bin/"
cp "$ROOT/src/kernel/sm120_format_probe.cu" "$ROOT/src/kernel/sm120_format_mma_probe.cu" "$OUT/source/"
cp "$ROOT/scripts/run_format_probe.py" "$ROOT/scripts/analyze_format_probe.py" "$ROOT/scripts/run_format_int_suite.sh" "$ROOT/CMakeLists.txt" "$OUT/source/"
git -C "$ROOT" diff --binary > "$OUT/source/working_tree.patch"
if ! git -C "$ROOT" ls-files --error-unmatch scripts/run_format_int_suite.sh >/dev/null 2>&1; then
  git -C "$ROOT" diff --no-index -- /dev/null scripts/run_format_int_suite.sh >> "$OUT/source/working_tree.patch" || test "$?" = 1
fi
git -C "$ROOT" status --short > "$OUT/source/git_status.txt"
nvidia-smi -q > "$OUT/gpu_start.txt"
# Validate the new packed-input paths on the original small full-check shape.
for p in int8 int4; do
  compute-sanitizer --tool memcheck --error-exitcode 91 \
    "$OUT/bin/hygemm_sm120_format_int_probe" --m=128 --n=16 --k=256 \
    --path="$p" --int-variant=2 --integer-compatible --full-check \
    --iterations=100 --repeats=3 > "$OUT/memcheck_${p}.log" 2>&1
done
"$PYTHON" "$ROOT/scripts/run_format_probe.py" --out "$OUT" \
  --binary "$OUT/bin/hygemm_sm120_format_probe" \
  --integer-binary "$OUT/bin/hygemm_sm120_format_int_probe" --integer-compatible
"$PYTHON" "$ROOT/scripts/run_format_probe.py" --out "$OUT" \
  --binary "$OUT/bin/hygemm_sm120_format_probe_tuned" --tuned --integer-compatible
nvidia-smi -q > "$OUT/gpu_before_mma.txt"
"$OUT/bin/hygemm_sm120_format_mma_probe" > "$OUT/mma.jsonl"
nvidia-smi -q > "$OUT/gpu_after_mma.txt"
cuobjdump --dump-sass "$OUT/bin/hygemm_sm120_format_probe" > "$OUT/gemm.sass"
cuobjdump --dump-sass "$OUT/bin/hygemm_sm120_format_probe_tuned" > "$OUT/gemm_tuned.sass"
cuobjdump --dump-sass "$OUT/bin/hygemm_sm120_format_int_probe" > "$OUT/gemm_int.sass"
cuobjdump --dump-sass "$OUT/bin/hygemm_sm120_format_mma_probe" > "$OUT/mma.sass"
"$PYTHON" "$ROOT/scripts/analyze_format_probe.py" --out "$OUT"
date -Is > "$OUT/completed.txt"
