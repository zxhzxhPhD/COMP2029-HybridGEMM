from __future__ import annotations

import unittest

import torch

from run_weight_error_analysis import (
    analyze_layer,
    resolve_budgets,
    sample_layers_by_module,
)


class WeightErrorAnalysisTest(unittest.TestCase):
    def test_module_sampling_covers_early_middle_late_per_module(self) -> None:
        names = [
            f"model.layers.{layer}.{module}"
            for layer in range(6)
            for module in ("self_attn.q_proj", "mlp.down_proj")
        ] + ["lm_head"]
        selected, indices = sample_layers_by_module(
            names, None, 3, "early_middle_late", seed=0
        )
        self.assertEqual(indices["self_attn.q_proj"], [0, 2, 5])
        self.assertEqual(indices["mlp.down_proj"], [0, 2, 5])
        self.assertEqual(len(selected), 6)
        self.assertNotIn("lm_head", selected)

    def test_budget_fractions_include_extreme_low_levels(self) -> None:
        self.assertEqual(resolve_budgets(100, [0.01, 0.02, 0.04], []), [0, 1, 2, 4, 100])
        self.assertEqual(resolve_budgets(17, [0.01], [2]), [0, 1, 2, 17])

    def test_full_promotion_reaches_w8a4_endpoint(self) -> None:
        torch.manual_seed(7)
        w = torch.randn(19, 96)
        x = torch.randn(96, 5)
        result = analyze_layer(
            w, x, [0.0, 0.01, 1.0], [],
            ["error_oracle", "activation_aware"], torch.device("cpu")
        )
        self.assertEqual(result["num_atoms"], 4)
        for method in result["methods"].values():
            self.assertEqual(sorted(method["ranking_atom_ids"]), list(range(4)))
            points = method["points"]
            self.assertAlmostEqual(
                points[-1]["relative_error"], result["w8a4_relative_error"], places=5
            )
            self.assertAlmostEqual(points[-1]["gap_recovery"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
