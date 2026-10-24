#!/usr/bin/env python3
"""Generate compact measurement tables, SASS evidence and a static research plot."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    base = json.loads((out / "summary.json").read_text())
    summary = json.loads((out / "summary_tuned.json").read_text())
    raw = [json.loads(l) for name in ["results.jsonl", "results_tuned.jsonl"]
           for l in (out / name).read_text().splitlines()]
    mma = [json.loads(l) for l in (out / "mma.jsonl").read_text().splitlines()]
    shapes = list(dict.fromkeys((r["m_out"], r["n_tokens"], r["k_in"]) for r in summary))
    lookup = {(*[r[k] for k in ["m_out", "n_tokens", "k_in"]], r["path"]): r for r in summary}
    paths = ["fp16", "w4a8", "w8a8_mx", "w4a4", "fp8_lt"]
    labels = ["FP16 cuBLASLt", "MX W4A8", "MX W8A8", "MX W4A4", "FP8 cuBLASLt"]
    lines = ["| tokens / out / in | " + " | ".join(labels) + " |", "|---|" + "---:|" * 5]
    for m, n, k in shapes:
        cells = []
        for p in paths:
            r = lookup[(m, n, k, p)]
            cells.append(f'{r["median_us"]:.3f} μs ({r["speedup_vs_fp16"]:.2f}×)')
        lines.append(f"| {n} / {m} / {k} | " + " | ".join(cells) + " |")
    lines += ["", "Selected configurations:", "", "| tokens / out / in | path | tile | schedule | TFLOP/s |", "|---|---|---|---|---:|"]
    for r in summary:
        if "tile" in r:
            lines.append(f'| {r["n_tokens"]} / {r["m_out"]} / {r["k_in"]} | {r["path"]} | {r["tile"]} | {r.get("schedule", "cooperative")} | {r["tflops"]:.1f} |')
    lines += ["", "Register-resident MMA (not GEMM):", "", "| path | K | TFLOP/s | relative to FP16 |", "|---|---:|---:|---:|"]
    for r in mma:
        lines.append(f'| {r["path"]} | {r["mma_k"]} | {r["tflops"]:.1f} | {r["tflops"]/mma[0]["tflops"]:.2f}× |')
    lines += ["", "Resources:", "", "| path | tile | schedule | SMEM bytes | registers/thread | stages | shared operand bits |", "|---|---|---|---:|---:|---:|---|"]
    seen = set()
    for r in raw:
        if "tile" not in r:
            continue
        key = (r["path"], tuple(r["tile"]), r.get("schedule", "cooperative"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f'| {key[0]} | {list(key[1])} | {key[2]} | {r["shared_bytes"]} | {r["registers_per_thread"]} | {r["stages"]} | {r["smem_operand_bits"]} |')
    spread = max((max(r["samples_us"]) - min(r["samples_us"])) / r["median_us"] for r in raw)
    lines += ["", f"Configurations: {len(raw)}. Largest within-configuration (max−min)/median: {spread:.2%}.",
              f'Total sampled output comparisons: {sum(r["checked"] for r in raw)}; failures: {sum(r["failures"] for r in raw)}.']
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    evidence = []
    for filename in ["gemm.sass", "gemm_tuned.sass", "mma.sass"]:
        for section in re.split(r"Function : ", (out / filename).read_text())[1:]:
            inst = [x for x in section.splitlines()[1:] if re.match(r"\s*/\*[0-9a-f]+\*/", x)]
            tensor = Counter()
            for x in inst:
                match = re.search(r"\b((?:Q|O|H)MMA\.[A-Z0-9.]+)", x)
                if match:
                    tensor[match[1]] += 1
            evidence.append({"file": filename, "function": section.splitlines()[0],
                             "static_mma_counts": dict(tensor),
                             "static_ldsm_unpack_count": sum("LDSM.U4x16P64TO8" in x for x in inst),
                             "static_shift_2_count": sum(bool(re.search(r" SHF.L.U32 .*0x2,", x)) for x in inst),
                             "examples": [x.strip() for x in inst if re.search(r"(?:QMMA|OMMA|HMMA)|LDSM.U4x16P64TO8| SHF.L.U32 .*0x2,", x)][:12]})
    (out / "sass_summary.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.3), gridspec_kw={"width_ratios": [2.4, 1]})
    colors = ["#167c80", "#e9a23b", "#6555aa", "#9aa2ab"]
    x = np.arange(len(shapes))
    for j, (p, label, color) in enumerate(zip(paths[1:], labels[1:], colors)):
        ys = [lookup[(*s, p)]["speedup_vs_fp16"] for s in shapes]
        bars = ax.bar(x + (j - 1.5) * .19, ys, .18, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.axhline(1, color="#333333", lw=1, linestyle="--", label="FP16 baseline")
    ax.set_xticks(x, [f"{n}\n{m}/{k}" for m, n, k in shapes])
    ax.set_xlabel("Tokens (top); output / input channels (bottom)")
    ax.set_ylabel("Speedup vs cuBLASLt FP16")
    ax.set_title("Prepacked GEMM: best of four fixed CUTLASS variants", loc="left", fontsize=12)
    ax.set_ylim(0, max(r["speedup_vs_fp16"] for r in summary) * 1.2)
    ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    bars = ax2.bar(["FP16", "W4A8", "W8A8", "W4A4"], [r["tflops"] for r in mma], color=["#4b5563", *colors[:3]])
    ax2.bar_label(bars, fmt="%.0f", fontsize=10, padding=3)
    ax2.set_ylabel("TFLOP/s")
    ax2.set_ylim(0, max(r["tflops"] for r in mma) * 1.2)
    ax2.set_title("Register-resident MMA diagnostic", loc="left", fontsize=12)
    fig.suptitle("RTX 5090 / SM120 | FP32 accumulation, directional measurements", fontsize=15, x=.06, ha="left")
    fig.text(.06, .015, "GEMM: FP16 output, hot-cache CUDA Graph, 3 x 100 iterations. MX formats use K32 UE8M0 scales. Quantization / packing excluded.", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(out / "speedups.png", dpi=180)
    fig.savefig(out / "speedups.svg")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
