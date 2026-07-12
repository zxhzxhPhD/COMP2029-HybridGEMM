"""Small-sample MXFP4 -> W4A8 precision-promotion analysis.

The analysis deliberately stays narrow: it reuses one captured snapshot,
samples early/middle/late linear modules, ranks K32 atoms with a global-error
oracle, and reports only relative error and gap recovery.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from quantize import MX_BLOCK_SIZE, quantize_w4a4, quantize_w4a8


DEFAULT_SNAPSHOT = Path(r"D:\User\document\repos\tmp\20260711_1605_qwen3_snapshot\snapshot.pt")
RESULT_ROOT = Path(r"D:\User\document\repos\tmp")
DEFAULT_BUDGETS = (0, 1, 2, 4, 8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument(
        "--layer-names",
        nargs="*",
        help="Explicit layer names. If omitted, sample early/middle/late modules.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="Absolute number of promoted K32 atoms.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def evenly_spaced_sample(names: list[str], count: int) -> list[str]:
    """Select deterministic early/middle/late positions from transformer linears."""
    candidates = [name for name in names if name.startswith("model.layers.")]
    if not candidates:
        candidates = names
    if count <= 0 or count >= len(candidates):
        return candidates
    positions = [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
    return [candidates[i] for i in positions]


def relative_error(reference: torch.Tensor, output: torch.Tensor) -> float:
    return (torch.linalg.vector_norm(reference - output) /
            torch.linalg.vector_norm(reference).clamp_min(1e-12)).item()


@torch.no_grad()
def analyze_layer(
    weight: torch.Tensor,
    activation: torch.Tensor,
    budgets: list[int],
    device: torch.device,
) -> dict:
    # Snapshot tensors are dequantized FP8 model values; they are the available
    # high-precision reference for this experiment.
    w = weight.float().to(device)
    x = activation.float().to(device)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D activation, got shape {tuple(x.shape)}")
    if w.shape[1] != x.shape[0]:
        raise ValueError(f"GEMM shape mismatch: {tuple(w.shape)} x {tuple(x.shape)}")

    w4, x4 = quantize_w4a4(w, x)
    # Weight is intentionally identical in both paths.
    _, x8 = quantize_w4a8(w, x)

    reference = w @ x
    baseline = w4 @ x4
    oracle = w4 @ x8
    base_residual = reference - baseline
    base_sq = torch.sum(base_residual * base_residual)
    oracle_sq = torch.sum((reference - oracle) ** 2)
    gap = torch.sqrt(base_sq) - torch.sqrt(oracle_sq)

    num_atoms = math.ceil(w.shape[1] / MX_BLOCK_SIZE)
    gains: list[tuple[int, float]] = []
    deltas: list[torch.Tensor] = []
    for atom in range(num_atoms):
        ks = atom * MX_BLOCK_SIZE
        ke = min(ks + MX_BLOCK_SIZE, w.shape[1])
        # delta is the output change when this K32 atom is promoted.
        delta = w4[:, ks:ke] @ (x8[ks:ke, :] - x4[ks:ke, :])
        gain = 2.0 * torch.sum(base_residual * delta) - torch.sum(delta * delta)
        gains.append((atom, gain.item()))
        deltas.append(delta)

    ranking = [atom for atom, _ in sorted(gains, key=lambda pair: pair[1], reverse=True)]
    requested = sorted(set(max(0, min(int(b), num_atoms)) for b in budgets) | {0, num_atoms})
    points = []
    hybrid = baseline.clone()
    promoted = 0
    for budget in requested:
        while promoted < budget:
            hybrid.add_(deltas[ranking[promoted]])
            promoted += 1
        hybrid_norm = torch.linalg.vector_norm(reference - hybrid)
        recovery = ((torch.sqrt(base_sq) - hybrid_norm) / gap).item() if abs(gap.item()) > 1e-12 else float("nan")
        points.append({
            "k32_atoms": budget,
            "atom_fraction": budget / num_atoms,
            "relative_error": relative_error(reference, hybrid),
            "gap_recovery": recovery,
        })

    return {
        "shape": [w.shape[0], w.shape[1], x.shape[1]],
        "num_k32_atoms": num_atoms,
        "reference_norm": torch.linalg.vector_norm(reference).item(),
        "w4a4_relative_error": relative_error(reference, baseline),
        "w4a8_relative_error": relative_error(reference, oracle),
        "w4a4_to_w4a8_relative_gap": (
            torch.linalg.vector_norm(baseline - oracle) /
            torch.linalg.vector_norm(reference).clamp_min(1e-12)
        ).item(),
        "points": points,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    print(f"Loading snapshot: {args.snapshot}")
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=False)

    available = [
        name for name in snapshot["layer_names"]
        if name in snapshot["weights"] and name in snapshot["activations"]
    ]
    selected = args.layer_names or evenly_spaced_sample(available, args.sample_count)
    missing = [name for name in selected if name not in available]
    if missing:
        raise KeyError(f"Layers not present in snapshot: {missing}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = RESULT_ROOT / f"{timestamp}_mxfp4_k32_sampled_analysis"
    result_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "snapshot": str(args.snapshot),
        "device": str(device),
        "quantization": "MXFP4 W/A baseline; MXFP4 W + MXFP8-E4M3 A promotion; UE8M0 K32 scales",
        "ranking": "per-K32 global squared-error reduction oracle",
        "selected_layers": selected,
        "budgets_k32_atoms": args.budgets,
        "layers": {},
    }

    print(f"Selected {len(selected)} layers:")
    for index, name in enumerate(selected, 1):
        print(f"  [{index}/{len(selected)}] {name}")
        report["layers"][name] = analyze_layer(
            snapshot["weights"][name], snapshot["activations"][name], args.budgets, device
        )
        layer = report["layers"][name]
        print(
            f"      relerr W4A4={layer['w4a4_relative_error']:.6f}, "
            f"W4A8={layer['w4a8_relative_error']:.6f}"
        )

    report_path = result_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
