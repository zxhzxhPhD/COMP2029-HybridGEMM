"""MMA-tile-granularity MXFP4 -> W8A4 weight-promotion analysis.

Weights are divided into M16 x K64 atoms.  Promoting an atom changes both of
its K32 MX blocks from E2M1 to E4M3 for every use of that weight.  Activations
remain MXFP4 in the baseline, promoted, and all-high-precision endpoints.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import platform
import random
import re
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from quantize import quantize_w4a4, quantize_w8a4


DEFAULT_SNAPSHOT = Path(r"D:\User\document\repos\tmp\20260711_1605_qwen3_snapshot\snapshot.pt")
RESULT_ROOT = Path(r"D:\User\document\repos\tmp")
DEFAULT_BUDGET_FRACTIONS = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0)
DEFAULT_METHODS = ("error_oracle", "activation_aware")
MMA_M = 16
MMA_K = 64
CUTLASS_REVISION = "e6233cba"
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze M16xK64 weight-tile promotion from W4A4 to W8A4."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--layer-names",
        nargs="*",
        help="Explicit full layer names; overrides module-based sampling.",
    )
    parser.add_argument(
        "--module-names",
        nargs="*",
        help=(
            "Module suffixes or glob patterns, for example self_attn.q_proj or "
            "mlp.*. Omit to sample every repeated transformer module type."
        ),
    )
    parser.add_argument("--layers-per-module", type=int, default=3)
    parser.add_argument(
        "--sampling",
        choices=("early_middle_late", "random"),
        default="early_middle_late",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS)
    )
    parser.add_argument(
        "--budget-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_BUDGET_FRACTIONS),
        help="Fractions of all M16xK64 weight atoms to promote.",
    )
    parser.add_argument(
        "--budget-atoms",
        type=int,
        nargs="*",
        default=(),
        help="Optional absolute atom budgets, unioned with fraction budgets.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--experiment-name", default="w8a4_m16k64_weight_analysis")
    return parser.parse_args()


def repeated_module_groups(names: list[str]) -> dict[str, list[tuple[int, str]]]:
    """Group transformer layer names by their suffix after the block index."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for name in names:
        match = LAYER_RE.match(name)
        if match:
            groups.setdefault(match.group(2), []).append((int(match.group(1)), name))
    for values in groups.values():
        values.sort()
    return groups


def _choose_group_layers(
    values: list[tuple[int, str]], count: int, sampling: str, rng: random.Random
) -> list[tuple[int, str]]:
    if count <= 0:
        return []
    if count >= len(values):
        return values
    if sampling == "random":
        return sorted(rng.sample(values, count))
    # Deterministic endpoints and evenly spaced interior positions give the
    # requested early/middle/late coverage when count == 3.
    positions = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
    return [values[position] for position in positions]


def sample_layers_by_module(
    names: list[str],
    module_patterns: list[str] | None,
    layers_per_module: int,
    sampling: str,
    seed: int,
) -> tuple[list[str], dict[str, list[int]]]:
    groups = repeated_module_groups(names)
    patterns = module_patterns or ["*"]
    selected_modules = [
        module for module in sorted(groups)
        if any(fnmatch.fnmatchcase(module, pattern) for pattern in patterns)
    ]
    unmatched = [
        pattern for pattern in patterns
        if not any(fnmatch.fnmatchcase(module, pattern) for module in groups)
    ]
    if unmatched:
        raise ValueError(f"Module patterns matched no repeated modules: {unmatched}")

    rng = random.Random(seed)
    selected: list[tuple[int, str, str]] = []
    sampled_indices: dict[str, list[int]] = {}
    for module in selected_modules:
        choices = _choose_group_layers(groups[module], layers_per_module, sampling, rng)
        sampled_indices[module] = [index for index, _ in choices]
        selected.extend((index, module, name) for index, name in choices)
    selected.sort(key=lambda item: (item[0], item[1]))
    return [name for _, _, name in selected], sampled_indices


def relative_error(reference: torch.Tensor, output: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(reference - output)
        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    ).item()


def resolve_budgets(
    num_atoms: int, fractions: list[float], absolute: list[int] | tuple[int, ...]
) -> list[int]:
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in fractions):
        raise ValueError("Budget fractions must be finite values in [0, 1]")
    budgets = {0, num_atoms}
    budgets.update(min(num_atoms, math.ceil(value * num_atoms)) for value in fractions)
    budgets.update(max(0, min(num_atoms, int(value))) for value in absolute)
    return sorted(budgets)


def _padded_tile_deltas(
    w4: torch.Tensor, w8: torch.Tensor, x4: torch.Tensor
) -> torch.Tensor:
    """Return output deltas with shape [M-tile, K-tile, 16, N]."""
    pad_m = (-w4.shape[0]) % MMA_M
    pad_k = (-w4.shape[1]) % MMA_K
    dw = F.pad(w8 - w4, (0, pad_k, 0, pad_m))
    padded_x = F.pad(x4, (0, 0, 0, pad_k))
    tiled_dw = dw.reshape(-1, MMA_M, dw.shape[1] // MMA_K, MMA_K)
    tiled_x = padded_x.reshape(padded_x.shape[0] // MMA_K, MMA_K, padded_x.shape[1])
    return torch.einsum("aikq,kqn->akin", tiled_dw, tiled_x)


def _method_scores(
    method: str, deltas: torch.Tensor, baseline_residual: torch.Tensor
) -> torch.Tensor:
    delta_energy = torch.sum(deltas * deltas, dim=(2, 3))
    if method == "activation_aware":
        # Actual output-change energy naturally weights the W8-vs-W4 weight
        # correction by the captured MXFP4 activations that consume each K64.
        return delta_energy
    if method == "error_oracle":
        residual_tiles = baseline_residual.reshape(deltas.shape[0], MMA_M, -1)
        alignment = torch.einsum("ain,akin->ak", residual_tiles, deltas)
        return 2.0 * alignment - delta_energy
    raise ValueError(f"Unsupported calibration method: {method}")


def _evaluate_ranking(
    ranking: torch.Tensor,
    deltas: torch.Tensor,
    reference: torch.Tensor,
    baseline: torch.Tensor,
    oracle_error_norm: torch.Tensor,
    budgets: list[int],
    original_m: int,
) -> list[dict]:
    residual = F.pad(reference - baseline, (0, 0, 0, (-original_m) % MMA_M))
    baseline_error_norm = torch.linalg.vector_norm(residual)
    gap = baseline_error_norm - oracle_error_norm
    correction = torch.zeros_like(residual).reshape(deltas.shape[0], MMA_M, -1)
    flat_deltas = deltas.reshape(-1, MMA_M, deltas.shape[-1])
    k_tiles = deltas.shape[1]
    promoted = 0
    points: list[dict] = []
    for budget in budgets:
        if budget > promoted:
            new_indices = ranking[promoted:budget]
            m_indices = torch.div(new_indices, k_tiles, rounding_mode="floor")
            correction.index_add_(0, m_indices, flat_deltas[new_indices])
            promoted = budget
        error_norm = torch.linalg.vector_norm(residual - correction.reshape_as(residual))
        recovery = ((baseline_error_norm - error_norm) / gap).item() if abs(gap.item()) > 1e-12 else float("nan")
        hybrid = reference - (residual - correction.reshape_as(residual))[:original_m]
        points.append({
            "promoted_atoms": budget,
            "atom_fraction": budget / ranking.numel(),
            "relative_error": relative_error(reference, hybrid),
            "gap_recovery": recovery,
        })
    return points


@torch.no_grad()
def analyze_layer(
    weight: torch.Tensor,
    activation: torch.Tensor,
    budget_fractions: list[float],
    budget_atoms: list[int] | tuple[int, ...],
    methods: list[str],
    device: torch.device,
) -> dict:
    w = weight.float().to(device)
    x = activation.float().to(device)
    if w.ndim != 2 or x.ndim != 2:
        raise ValueError(f"Expected 2D weight and activation, got {tuple(w.shape)}, {tuple(x.shape)}")
    if w.shape[1] != x.shape[0]:
        raise ValueError(f"GEMM shape mismatch: {tuple(w.shape)} x {tuple(x.shape)}")

    w4, x4 = quantize_w4a4(w, x)
    w8, x4_check = quantize_w8a4(w, x)
    if not torch.equal(x4, x4_check):
        raise RuntimeError("Activation quantization changed between W4A4 and W8A4 paths")

    reference = w @ x
    baseline = w4 @ x4
    oracle = w8 @ x4
    deltas = _padded_tile_deltas(w4, w8, x4)
    num_atoms = deltas.shape[0] * deltas.shape[1]
    budgets = resolve_budgets(num_atoms, budget_fractions, budget_atoms)
    padded_residual = F.pad(reference - baseline, (0, 0, 0, (-w.shape[0]) % MMA_M))
    oracle_error_norm = torch.linalg.vector_norm(reference - oracle)

    curves: dict[str, dict] = {}
    for method in methods:
        scores = _method_scores(method, deltas, padded_residual)
        ranking = torch.argsort(scores.reshape(-1), descending=True, stable=True)
        curves[method] = {
            "ranking_atom_ids": ranking.cpu().tolist(),
            "points": _evaluate_ranking(
                ranking, deltas, reference, baseline, oracle_error_norm, budgets, w.shape[0]
            ),
        }

    return {
        "shape_mkn": [w.shape[0], w.shape[1], x.shape[1]],
        "mma_atom_shape_mk": [MMA_M, MMA_K],
        "num_m_tiles": deltas.shape[0],
        "num_k64_tiles": deltas.shape[1],
        "num_atoms": num_atoms,
        "budgets_atoms": budgets,
        "reference_norm": torch.linalg.vector_norm(reference).item(),
        "w4a4_relative_error": relative_error(reference, baseline),
        "w8a4_relative_error": relative_error(reference, oracle),
        "w4a4_to_w8a4_relative_gap": (
            torch.linalg.vector_norm(baseline - oracle)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        ).item(),
        "methods": curves,
    }


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    args = parse_args()
    if args.layers_per_module < 1:
        raise ValueError("--layers-per-module must be positive")
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
    if args.layer_names:
        selected = args.layer_names
        sampled_indices = None
    else:
        selected, sampled_indices = sample_layers_by_module(
            available, args.module_names, args.layers_per_module, args.sampling, args.seed
        )
    missing = [name for name in selected if name not in available]
    if missing:
        raise KeyError(f"Layers not present in snapshot: {missing}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = RESULT_ROOT / f"{timestamp}_{args.experiment_name}"
    result_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "purpose": "M16xK64 weight-tile W8A4 promotion analysis with fixed MXFP4 activations",
        "snapshot": str(args.snapshot),
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "revisions": {
            "hygemm_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "cutlass": CUTLASS_REVISION,
        },
        "quantization": "MXFP4 activation fixed; weight atoms promoted from MXFP4 to MXFP8-E4M3; UE8M0 K32 scales",
        "atom": "one M16xK64 weight tile; promotion upgrades its paired K32 weight blocks",
        "atom_indexing": "row-major: atom_id = m_tile * num_k64_tiles + k64_tile",
        "calibration_methods": {
            "error_oracle": "baseline global squared-error reduction from promoting one M16xK64 atom",
            "activation_aware": "squared output-change energy of each W8-vs-W4 atom under captured MXFP4 activations",
        },
        "sampling": {
            "mode": "explicit" if args.layer_names else args.sampling,
            "module_patterns": args.module_names,
            "layers_per_module": args.layers_per_module,
            "seed": args.seed,
            "sampled_layer_indices": sampled_indices,
        },
        "selected_layers": selected,
        "requested_budget_fractions": args.budget_fractions,
        "requested_budget_atoms": args.budget_atoms,
        "methods": args.methods,
        "layers": {},
    }

    print(f"Output: {result_dir}")
    print(f"Selected {len(selected)} layers on {device}:")
    for index, name in enumerate(selected, 1):
        start = time.perf_counter()
        print(f"  [{index}/{len(selected)}] {name}", flush=True)
        report["layers"][name] = analyze_layer(
            snapshot["weights"][name], snapshot["activations"][name],
            args.budget_fractions, args.budget_atoms, args.methods, device
        )
        layer = report["layers"][name]
        print(
            f"      atoms={layer['num_atoms']}, W4A4={layer['w4a4_relative_error']:.6f}, "
            f"W8A4={layer['w8a4_relative_error']:.6f}, elapsed={time.perf_counter() - start:.1f}s",
            flush=True,
        )
        # Checkpoint after every layer so a long local run remains inspectable.
        (result_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
        )
    print(f"Report: {result_dir / 'report.json'}")


if __name__ == "__main__":
    main()
