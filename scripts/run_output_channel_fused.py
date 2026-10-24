#!/usr/bin/env python3
"""Fused precision-switch validation and matched replay of optimized dual plans."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--dual-binary", type=Path, required=True)
    ap.add_argument("--dual-plans", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phase", choices=["all", "validate", "tune"], default="all")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    root = Path(__file__).resolve().parents[1]
    meta = {
        "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": capture(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.limit", "--format=csv"]),
        "nvcc": capture(["/usr/local/cuda/bin/nvcc", "--version"]),
        "base_commit": capture(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "cutlass_commit": capture(["git", "-C", "/root/code/cutlass", "rev-parse", "HEAD"]),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root / "src/kernel").glob("output_channel*"))},
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "dual_binary_sha256": hashlib.sha256(args.dual_binary.read_bytes()).hexdigest(),
        "protocol": "Common tile/stages/schedule for both precisions; 8 fused candidates; prepacked inputs; fused scatter; graph timing; alternating matched dual/fused measurements, 5x100 requests. Diagnostics disabled during timing. Existing dual plans replayed on identical inputs; previously unmeasured shapes obtain a plan from the original dual executable.",
    }
    (args.out / f"environment_{args.phase}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def run(name, flags, *, dual=False, sanitizer=None):
        cmd = [str(args.dual_binary if dual else args.binary), *flags]
        if sanitizer:
            cmd = ["/usr/local/cuda/bin/compute-sanitizer", "--tool", sanitizer, "--error-exitcode", "99", *cmd]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              env=env, timeout=300 if sanitizer else 120)
        (args.out / f"{name}.log").write_text("$ " + " ".join(cmd) + "\n" + proc.stdout, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"{name}: {proc.stdout[-4000:]}")
        rows = [json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{"kind"')]
        (args.out / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        final = rows[-1]
        print(name, final.get("fused_us", final.get("median_us", "passed")),
              "speedup", final.get("speedup_vs_dual", "-"), flush=True)
        return final

    if args.phase in ["all", "validate"]:
        cases = [(257, 7, 256, h, "random") for h in [0, 1, 26, 128, 256, 257]]
        cases += [(511, 33, 384, 73, "interleaved"), (129, 1, 128, 17, "random"), (255, 17, 128, 51, "contiguous")]
        vals = []
        for o, t, k, h, mapping in cases:
            vals.append(run(f"check_{o}_{t}_{k}_{h}", [f"--out={o}", f"--tokens={t}", f"--k={k}",
                f"--high={h}", f"--mapping={mapping}", "--full-check", "--check-only", "--audit"]))
        for k in [128, 384, 1024]:
            vals.append(run(f"switch_stress_{k}", ["--out=4096", "--tokens=2049", f"--k={k}", "--high=2048",
                "--mapping=random", "--check-only", "--audit", "--require-switch"]))
        (args.out / "validation_summary.json").write_text(json.dumps(vals, indent=2), encoding="utf-8")
        flags = ["--out=257", "--tokens=7", "--k=384", "--high=26", "--mapping=random", "--full-check", "--check-only", "--audit"]
        run("memcheck", flags, sanitizer="memcheck")
        run("synccheck", flags, sanitizer="synccheck")
        for variant in [6, 7]:
            switch_flags = ["--out=4096", "--tokens=2049", "--k=384", "--high=2048", "--mapping=random",
                            "--check-only", "--audit", "--require-switch", f"--fused-id={variant}"]
            run(f"switch_memcheck_{variant}", switch_flags, sanitizer="memcheck")
            run(f"switch_synccheck_{variant}", switch_flags, sanitizer="synccheck")

    if args.phase in ["all", "tune"]:
        plans = json.loads(args.dual_plans.read_text())
        shapes = [(4096, t, 4096, 410) for t in [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]]
        shapes += [(14336, 128, 4096, 1434), (4096, 128, 14336, 410)]
        shapes += [(4096, t, 4096, h) for t in [16, 128] for h in [0, 4096]]
        shapes += [(4096, 128, 4096, 2048)]
        results = []
        for o, t, k, h in shapes:
            key = {"out": o, "tokens": t, "k": k, "high": h}
            plan = next((p for p in plans if all(p[x] == v for x, v in key.items())), None)
            shape_flags = [f"--out={o}", f"--tokens={t}", f"--k={k}", f"--high={h}"]
            if plan is None:
                plan = run(f"dual_tune_{o}_{t}_{k}_{h}", [*shape_flags, "--no-baseline"], dual=True)
                plans.append(plan)
            result = run(f"compare_{o}_{t}_{k}_{h}", [*shape_flags, f"--low-id={plan['low_id']}",
                         f"--high-id={plan['high_id']}", f"--policy={plan['policy']}", "--repeats=5",
                         *(["--measure-eager"] if t <= 128 and 0 < h < o else [])])
            results.append(result)
            (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        fields = ["out", "tokens", "k", "high", "fused_id", "fused_us", "dual_us", "speedup_vs_dual", "low_id", "high_id", "dual_policy"]
        with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        (args.out / "dual_plans_used.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")
        (args.out / "fused_plans.json").write_text(json.dumps([
            {x: r[x] for x in ["out", "tokens", "k", "high", "mapping", "fused_id"]} for r in results], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
