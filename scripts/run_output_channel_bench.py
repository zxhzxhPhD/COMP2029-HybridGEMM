#!/usr/bin/env python3
"""Validate scatter and tune a bounded family of two-kernel output-channel GEMMs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def capture(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()


def gpu():
    return capture(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu", "--format=csv,noheader"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["validate", "tune", "all"], default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    root = Path(__file__).resolve().parents[1]
    metadata = {
        "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_start": gpu(),
        "nvcc": capture(["/usr/local/cuda/bin/nvcc", "--version"]),
        "base_commit": capture(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "cutlass_commit": capture(["git", "-C", "/root/code/cutlass", "rev-parse", "HEAD"]),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root / "src/kernel").glob("output_channel*"))},
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "protocol": "Prepacked MXFP4/MXFP8 weights, shared MXFP8 activation, K32 UE8M0 scales, FP32 accumulation, FP16 fused scatter. Exact logical high-channel count, no rounding to tile size. Hot-cache CUDA Graph; 3x100 requests after warmup. Top-3 per path then joint pair/schedule search; not exhaustive joint search.",
        "excluded": ["quantization", "offline weight gather", "activation pack", "model accuracy"],
        "mixllm": "Original implementation remains unchanged; no new integer implementation or integer benchmark in this run.",
    }
    (args.out / f"environment_{args.phase}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def run(name, options, sanitizer=False):
        directory = args.out / ("sanitizer" if sanitizer else ("validation" if name.startswith("check") else "measurements"))
        directory.mkdir(exist_ok=True)
        cmd = [str(args.binary), *options]
        if sanitizer:
            cmd = ["/usr/local/cuda/bin/compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99", *cmd]
        before = gpu()
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=600)
        (directory / f"{name}.log").write_text("$ " + " ".join(cmd) + "\n" + result.stdout, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}): {result.stdout[-4000:]}")
        rows = [json.loads(x) for x in result.stdout.splitlines() if x.startswith('{"kind"')]
        summary = next(x for x in rows if x["kind"] == "result")
        summary.update(name=name, gpu_before=before, gpu_after=gpu())
        (directory / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"{name}: {summary['median_us']:.3f} us, {summary['policy']}, ids={summary['low_id']}/{summary['high_id']}", flush=True)
        return summary

    if args.phase in ["all", "validate"]:
        results = []
        cases = [(257, 7, 256, h, "random") for h in [0, 1, 26, 128, 256, 257]]
        cases += [(511, 33, 384, 73, "interleaved"), (129, 1, 128, 17, "random"), (255, 17, 128, 51, "contiguous")]
        for out, tokens, k, high, mapping in cases:
            options = [f"--out={out}", f"--tokens={tokens}", f"--k={k}", f"--high={high}", f"--mapping={mapping}",
                       "--full-check", "--verify-all", "--no-baseline", "--low-id=0", "--high-id=0", "--iterations=3", "--repeats=1"]
            results.append(run(f"check_{out}_{tokens}_{k}_{high}_{mapping}", options))
        (args.out / "validation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        # Residues on all three axes, nontrivial scatter, both streams and every variant.
        run("memcheck_scatter", ["--out=257", "--tokens=7", "--k=384", "--high=26", "--mapping=random",
            "--full-check", "--verify-all", "--no-baseline", "--low-id=0", "--high-id=0", "--iterations=2", "--repeats=1"], sanitizer=True)

    if args.phase in ["all", "tune"]:
        shapes = [(4096, t, 4096, .1) for t in [1, 16, 128, 512, 2048]]
        shapes += [(14336, 128, 4096, .1), (4096, 128, 14336, .1)]
        shapes += [(4096, t, 4096, f) for t in [16, 128, 2048] for f in [0., 1.]]
        shapes += [(4096, 128, 4096, .5)]
        results = []
        for out, tokens, k, fraction in shapes:
            name = f"tune_{out}_{tokens}_{k}_{fraction:.2f}"
            results.append(run(name, [f"--out={out}", f"--tokens={tokens}", f"--k={k}", f"--fraction={fraction}"]))
            (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        fields = ["out", "tokens", "k", "high", "actual_fraction", "low_id", "high_id", "policy", "median_us", "fp16_us", "speedup_fp16", "tflops", "kernels_per_request"]
        with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        plans = [{k: r[k] for k in ["out", "tokens", "k", "high", "mapping", "low_id", "high_id", "policy"]} for r in results]
        (args.out / "tuning_plans.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")
        # Replay a stored plan with a different scatter order; tuning is not required for execution.
        r = next(r for r in results if r["tokens"] == 128 and r["k"] == 4096 and r["out"] == 4096 and 0 < r["high"] < 4096)
        run("replay_random_map", ["--out=4096", "--tokens=128", "--k=4096", f"--high={r['high']}", "--mapping=random",
            f"--low-id={r['low_id']}", f"--high-id={r['high_id']}", f"--policy={r['policy']}", "--no-baseline"])


if __name__ == "__main__":
    main()
