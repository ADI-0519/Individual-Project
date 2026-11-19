"""CPU-only checks of the validation tolerance and prespecified selection."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_full_baseline.py"
SPEC = importlib.util.spec_from_file_location("full_baseline_validator", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load standalone full-baseline validation module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullBaselineValidationTests(unittest.TestCase):
    def test_bf16_neighbors_include_asymmetric_spacing_negative_values_and_zero(self):
        reference = np.asarray([1.0, -1.0, 0.0, -0.0], dtype=np.float32)
        upper = np.asarray([1.0078125, -0.99609375, 0.0, 0.0], dtype=np.float32)
        lower = np.asarray([0.99609375, -1.0078125, 0.0, 0.0], dtype=np.float32)
        self.assertTrue(MODULE.bf16_neighbor_check(reference, upper)["passed"])
        self.assertTrue(MODULE.bf16_neighbor_check(reference, lower)["passed"])
        beyond = upper.copy()
        beyond[0] = 1.015625
        check = MODULE.bf16_neighbor_check(reference, beyond)
        self.assertFalse(check["passed"])
        self.assertEqual(check["outside_neighbor_count"], 1)
        smallest = np.asarray([0x00010000], dtype=np.uint32).view(np.float32)
        zeros = np.asarray([0.0], dtype=np.float32)
        self.assertTrue(MODULE.bf16_neighbor_check(zeros, smallest)["passed"])
        self.assertTrue(MODULE.bf16_neighbor_check(zeros, -smallest)["passed"])
        self.assertFalse(MODULE.bf16_neighbor_check(zeros, 2 * smallest)["passed"])

    def test_bf16_comparison_refuses_off_grid_nonfinite_and_mismatched_arrays(self):
        for reference, observed in (
            (np.asarray([1.001], dtype=np.float32), np.asarray([1.001], dtype=np.float32)),
            (np.asarray([1.0], dtype=np.float32), np.asarray([np.nan], dtype=np.float32)),
            (np.ones((1, 2), dtype=np.float32), np.ones((2,), dtype=np.float32)),
        ):
            with self.assertRaises(ValueError):
                MODULE.bf16_neighbor_check(reference, observed)

    def test_selection_is_fixed_by_sources_not_input_order_or_values(self):
        examples = [
            {"example_id": f"source-{index:03d}-{label}", "pair_id": f"source-{index:03d}"}
            for index in range(268)
            for label in (0, 1)
        ]
        selected = MODULE.select_replay_examples(examples)
        self.assertEqual(len(selected), 22)
        self.assertEqual(selected, MODULE.select_replay_examples(list(reversed(examples))))
        sources = {row["pair_id"] for row in selected}
        self.assertIn("source-000", sources)
        self.assertIn("source-267", sources)
        self.assertIn("source-027", sources)
        self.assertNotIn("source-026", sources)


if __name__ == "__main__":
    unittest.main()
