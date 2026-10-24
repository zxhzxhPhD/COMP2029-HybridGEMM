#!/usr/bin/env python3
"""Small SM120 format sweep. All generated artifacts go to the explicit --out."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def gpu():
    return command([
        "nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used",
        "--format=csv,noheader",
    ]).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tuned", action="store_true", help="Run the two additional tuned variants and merge with results.jsonl")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    root = Path(__file__).resolve().parents[1]
    metadata = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_start": gpu(),
        "gpu_details": command(["nvidia-smi", "-q"]),
        "nvcc": command(["/usr/local/cuda/bin/nvcc", "--version"]),
        "commit": command(["git", "-C", str(root), "rev-parse", "HEAD"]).strip(),
        "cutlass_commit": command(["git", "-C", "/root/code/cutlass", "rev-parse", "HEAD"]).strip(),
        "git_status": command(["git", "-C", str(root), "status", "--short"]),
        "source_sha256": hashlib.sha256((root / "src/kernel/sm120_format_probe.cu").read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "convention": "Y[M_out,N_tokens] = W[M_out,K_in] X[N_tokens,K_in]^T; column-major Y",
        "timing": "prepacked resident tensors; 100 sequential GEMMs per CUDA graph; 80 ms warmup per configuration; 3 measured graph replays; hot-cache; no clock locking",
        "low_precision": "MXFP4 E2M1 / MXFP8 E4M3, K32 UE8M0 scales; FP32 accumulator, FP16 output",
        "reference": "all paths use identical representable matrices; 256 sampled FP16 outputs checked exactly; full check on small smoke shape",
        "excluded": "quantization, online packing, output-channel split/scatter, model accuracy; no INT MixLLM comparison",
    }
    metadata["reduction_policy"] = "cuBLASLt split-K intermediate storage restricted to FP32 compute type"
    suffix = "_tuned" if args.tuned else ""
    (args.out / f"environment{suffix}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    configs = [(p, t) for p in ["w4a4", "w4a8", "w8a8_mx"] for t in [64, 128]] + [("fp16", 64), ("fp8_lt", 64)]
    if args.tuned:
        configs = [(p, t) for p in ["w4a4", "w4a8", "w8a8_mx"] for t in [16, 64]]

    def run(shape, config, smoke=False):
        m, n, k = shape
        path, tile = config
        cmd = [str(args.binary), f"--m={m}", f"--n={n}", f"--k={k}",
               f"--path={path}", f"--tile-n={tile}", "--iterations=100", "--repeats=3"]
        if smoke:
            cmd.append("--full-check")
        start = gpu()
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        stem = f"{'smoke' if smoke else 'bench'}{suffix}_{m}_{n}_{k}_{path}_{tile}"
        (args.out / f"{stem}.log").write_text("$ " + " ".join(cmd) + "\n" + proc.stdout, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"Failed {stem}: {proc.stdout}")
        row = json.loads(next(line for line in proc.stdout.splitlines() if line.startswith('{"path"')))
        assert row["failures"] == 0 and row["median_us"] > 0
        row.update(gpu_before=start, gpu_after=gpu(), smoke=smoke)
        print(stem, f'{row["median_us"]:.3f} us', flush=True)
        return row

    with (args.out / f"smoke{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for config in configs:
            f.write(json.dumps(run((128, 16, 256), config, True)) + "\n")
            f.flush()

    shapes = [(4096, 1, 4096), (4096, 16, 4096), (4096, 128, 4096),
              (4096, 512, 4096), (4096, 2048, 4096),
              (14336, 128, 4096), (4096, 128, 14336)]
    rows = []
    if args.tuned:
        rows = [json.loads(line) for line in (args.out / "results.jsonl").read_text().splitlines()]
    rng = random.Random(20261024)
    with (args.out / f"results{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for shape in shapes:
            order = list(configs)
            rng.shuffle(order)
            for config in order:
                row = run(shape, config)
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
    best = {}
    for row in rows:
        key = (row["m_out"], row["n_tokens"], row["k_in"], row["path"])
        if key not in best or row["median_us"] < best[key]["median_us"]:
            best[key] = row
    summary = []
    for shape in shapes:
        base = best[(*shape, "fp16")]["median_us"]
        for path in ["fp16", "w4a8", "w8a8_mx", "w4a4", "fp8_lt"]:
            row = dict(best[(*shape, path)])
            row["speedup_vs_fp16"] = base / row["median_us"]
            summary.append(row)
    (args.out / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = ["m_out", "n_tokens", "k_in", "path", "median_us", "tflops", "speedup_vs_fp16", "tile", "schedule", "samples_us"]
    with (args.out / f"summary{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    (args.out / "gpu_end.txt").write_text(gpu(), encoding="utf-8")


if __name__ == "__main__":
    main()
