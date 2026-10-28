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
    parser.add_argument("--plot-only", action="store_true", help="Regenerate the main plots without rewriting archived tables or SASS evidence")
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
    for path, label in [("int8", "INT8 x INT8"), ("int4", "INT4 x INT4 (PTX)")]:
        if any(r["path"] == path for r in summary):
            paths.append(path)
            labels.append(label)
    has_int = "int8" in paths
    lines = ["| tokens / out / in | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(paths)]
    for m, n, k in shapes:
        cells = []
        for p in paths:
            r = lookup[(m, n, k, p)]
            cells.append(f'{r["median_us"]:.3f} μs ({r["speedup_vs_fp16"]:.2f}×)')
        lines.append(f"| {n} / {m} / {k} | " + " | ".join(cells) + " |")
    lines += ["", "Selected configurations:", "", "| tokens / out / in | path | tile | schedule | TOp/s (FMA=2) |", "|---|---|---|---|---:|"]
    for r in summary:
        if "tile" in r:
            lines.append(f'| {r["n_tokens"]} / {r["m_out"]} / {r["k_in"]} | {r["path"]} | {r["tile"]} | {r.get("schedule", "cooperative")} | {r["tflops"]:.1f} |')
    lines += ["", "Register-resident MMA (not GEMM):", "", "| path | K | TOp/s (FMA=2) | relative to FP16 |", "|---|---:|---:|---:|"]
    for r in mma:
        lines.append(f'| {r["path"]} | {r["mma_k"]} | {r["tflops"]:.1f} | {r["tflops"]/mma[0]["tflops"]:.2f}× |')
    lines += ["", "Resources:", "", "| path | tile | schedule | SMEM bytes | registers/thread | stages | shared operand bits |", "|---|---|---|---:|---:|---:|---|"]
    seen = set()
    for r in raw:
        if "tile" not in r:
            continue
        key = (r["path"], tuple(r["tile"]), r.get("schedule", "cooperative"), r["stages"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f'| {key[0]} | {list(key[1])} | {key[2]} | {r["shared_bytes"]} | {r["registers_per_thread"]} | {r["stages"]} | {r["smem_operand_bits"]} |')
    spread = max((max(r["samples_us"]) - min(r["samples_us"])) / r["median_us"] for r in raw)
    lines += ["", f"Configurations: {len(raw)}. Largest within-configuration (max−min)/median: {spread:.2%}.",
              f'Total sampled output comparisons: {sum(r["checked"] for r in raw)}; failures: {sum(r["failures"] for r in raw)}.']
    if not args.plot_only:
        (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    evidence = []
    for filename in ["gemm.sass", "gemm_tuned.sass", "mma.sass"] + (["gemm_int.sass"] if has_int else []):
        for section in re.split(r"Function : ", (out / filename).read_text())[1:]:
            inst = [x for x in section.splitlines()[1:] if re.match(r"\s*/\*[0-9a-f]+\*/", x)]
            tensor = Counter()
            for x in inst:
                match = re.search(r"\b((?:Q|O|H|I)MMA\.[A-Z0-9.]+)", x)
                if match:
                    tensor[match[1]] += 1
            evidence.append({"file": filename, "function": section.splitlines()[0],
                             "static_mma_counts": dict(tensor),
                             "static_ldsm_unpack_count": sum("LDSM.U4x16P64TO8" in x for x in inst),
                             "static_shift_2_count": sum(bool(re.search(r" SHF.L.U32 .*0x2,", x)) for x in inst),
                             "examples": [x.strip() for x in inst if re.search(r"(?:QMMA|OMMA|HMMA|IMMA)|LDSM.U4x16P64TO8| SHF.L.U32 .*0x2,", x)][:12]})
    if not args.plot_only:
        (out / "sass_summary.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    # INT4 PTX lowers to INT8 instructions on SM120. Keep its measurements in
    # the full tables and raw archive, while plotting the native paths only.
    plot_paths = [p for p in paths if p != "int4"]
    plot_labels = [label for p, label in zip(paths, labels) if p != "int4"]
    plot_mma = [r for r in mma if r["path"] != "int4"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17.2, 5.8) if has_int else (13.5, 5.3), gridspec_kw={"width_ratios": [2.4, 1.25 if has_int else 1]})
    colors = ["#167c80", "#e9a23b", "#6555aa", "#9aa2ab", "#3979b5", "#c6644c"]
    x = np.arange(len(shapes))
    width = .78 / (len(plot_paths) - 1)
    for j, (p, label, color) in enumerate(zip(plot_paths[1:], plot_labels[1:], colors)):
        ys = [lookup[(*s, p)]["speedup_vs_fp16"] for s in shapes]
        bars = ax.bar(x + (j - (len(plot_paths) - 2) / 2) * width, ys, width * .94, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2, rotation=90 if has_int else 0)
    ax.axhline(1, color="#333333", lw=1, linestyle="--", label="FP16 baseline")
    ax.set_xticks(x, [f"{n}\n{m}/{k}" for m, n, k in shapes])
    ax.set_xlabel("Tokens (top); output / input channels (bottom)")
    ax.set_ylabel("Speedup vs cuBLASLt FP16")
    ax.set_title("Prepacked GEMM: best of four fixed CUTLASS variants per path", loc="left", fontsize=12)
    ax.set_ylim(0, max(r["speedup_vs_fp16"] for r in summary if r["path"] in plot_paths) * 1.28)
    ax.legend(loc="upper left", ncol=3 if has_int else 2, fontsize=8, frameon=False)
    mma_labels = {"fp16":"FP16", "w4a8":"MX\nW4A8", "w8a8_mx":"MX\nW8A8", "w4a4":"MX\nW4A4", "int8":"INT8", "int4":"INT4\n(PTX)"}
    mma_colors = {"fp16":"#4b5563", **dict(zip(plot_paths[1:], colors))}
    bars = ax2.bar([mma_labels[r["path"]] for r in plot_mma], [r["tflops"] for r in plot_mma], color=[mma_colors[r["path"]] for r in plot_mma])
    ax2.bar_label(bars, fmt="%.0f", fontsize=10, padding=3)
    ax2.set_ylabel("TOp/s (multiply-add = 2 ops)" if has_int else "TFLOP/s")
    ax2.set_ylim(0, max(r["tflops"] for r in plot_mma) * 1.2)
    ax2.set_title("Register-resident MMA diagnostic", loc="left", fontsize=12)
    fig.suptitle("RTX 5090 / SM120 | " + ("FP32 / INT32 accumulation" if has_int else "FP32 accumulation") + ", directional measurements", fontsize=15, x=.06, ha="left")
    fig.text(.06, .04 if has_int else .015, "GEMM: FP16 output, hot-cache CUDA Graph, 3 x 100 iterations. MX formats use K32 UE8M0 scales. Quantization / packing excluded.", fontsize=9, color="#555555")
    if has_int:
        fig.text(.06, .01, "All GEMMs use identical exactly representable inputs. Floating point: FP32 accumulation; INT8: INT32 accumulation.", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .08 if has_int else .05, 1, .94))
    fig.savefig(out / "speedups.png", dpi=180)
    fig.savefig(out / "speedups.svg")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
