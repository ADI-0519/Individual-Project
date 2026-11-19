"""Numerical and provenance contracts for the token-trained Apollo adaptation."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from latent_probes.apollo_probe import (
    INSTRUCTION_TARGET,
    calibrate_control_threshold,
    evaluate_apollo_scores,
    score_apollo_probe,
    train_apollo_probe,
)


class ApolloProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.signature = json.dumps({"checkpoint": "fixture", "layer": 8, "precision": "fixture"})

    def features(self, name, X, ids, signature=None):
        path = self.root / name
        np.savez_compressed(
            path,
            X=np.asarray(X, dtype=float),
            example_ids=np.asarray(ids, dtype=str),
            feature_signature=np.asarray(signature or self.signature),
        )
        return path

    def metadata(self, name, rows):
        path = self.root / name
        fields = ["example_id", "pair_id", "split", "label", "label_target", "label_audited"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def scores(self, name, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["example_id", "score"])
            writer.writeheader()
            writer.writerows(
                {"example_id": identifier, "score": value} for identifier, value in rows
            )
        return path

    def training(self):
        features = self.features(
            "train.npz",
            [[-3], [-1], [1], [3], [100], [102]],
            ["h", "h", "d", "d", "vh", "vd"],
        )
        metadata = self.metadata(
            "train.csv",
            [
                {
                    "example_id": identifier,
                    "pair_id": pair,
                    "split": split,
                    "label": label,
                    "label_target": INSTRUCTION_TARGET,
                }
                for identifier, pair, split, label in (
                    ("h", "train-pair", "train", 0),
                    ("d", "train-pair", "train", 1),
                    ("vh", "val-pair", "validation", 0),
                    ("vd", "val-pair", "validation", 1),
                )
            ],
        )
        return features, metadata

    def test_scaler_uses_training_tokens_and_numeric_artifact(self):
        features, metadata = self.training()
        report = train_apollo_probe(features, metadata, self.root / "probe")
        with np.load(self.root / "probe/probe.npz", allow_pickle=False) as saved:
            np.testing.assert_allclose(saved["mean"], [0])
            np.testing.assert_allclose(saved["scale"], [np.sqrt(5)])
            self.assertEqual(float(saved["intercept"]), 0)
            self.assertGreater(float(saved["coefficient"][0]), 0)
        self.assertEqual(report["settings"]["C"], 0.1)
        self.assertFalse(report["settings"]["fit_intercept"])
        self.assertEqual(report["n_training_token_rows"], 4)
        self.assertEqual(report["validation_diagnostic"]["roc_auc"], 1)

    def test_mean_raw_margin_not_mean_sigmoid(self):
        features, metadata = self.training()
        train_apollo_probe(features, metadata, self.root / "probe")
        test = self.features("score.npz", [[-20], [4], [4]], ["mixed", "mixed", "mixed"])
        output = self.root / "scores.csv"
        score_apollo_probe(test, self.root / "probe", output)
        with output.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        with np.load(self.root / "probe/probe.npz", allow_pickle=False) as saved:
            margins = (np.asarray([-20, 4, 4]) - saved["mean"][0]) / saved["scale"][0]
            margins = margins * saved["coefficient"][0]
        self.assertAlmostEqual(float(row["score"]), float(margins.mean()))
        self.assertLess(float(row["score"]), 0)
        self.assertEqual(int(row["n_tokens"]), 3)
        self.assertNotAlmostEqual(float(row["score"]), float((1 / (1 + np.exp(-margins))).mean()))

    def test_pair_leakage_and_missing_pair_class_are_rejected(self):
        features, metadata = self.training()
        original = metadata.read_text(encoding="utf-8")
        metadata.write_text(original.replace("val-pair", "train-pair"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "leakage"):
            train_apollo_probe(features, metadata, self.root / "leak")
        metadata.write_text(
            original.replace("d,train-pair,train,1", "d,train-pair,train,0"), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "contain labels"):
            train_apollo_probe(features, metadata, self.root / "one-class")

    def test_feature_identity_and_noncontiguous_token_ids_are_rejected(self):
        features, metadata = self.training()
        train_apollo_probe(features, metadata, self.root / "probe")
        wrong = self.features("wrong.npz", [[1]], ["new"], '{"checkpoint":"other"}')
        with self.assertRaisesRegex(ValueError, "identity"):
            score_apollo_probe(wrong, self.root / "probe", self.root / "wrong.csv")
        scattered = self.features("scattered.npz", [[1], [2], [3]], ["a", "b", "a"])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            score_apollo_probe(scattered, self.root / "probe", self.root / "scattered.csv")

    def test_validation_requires_complete_pairs_and_training_rejects_duplicate_condition(self):
        features, metadata = self.training()
        original = metadata.read_text(encoding="utf-8")
        metadata.write_text(
            original.replace("vd,val-pair,validation,1", "vd,val-pair,validation,0"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Validation pair.*exactly once"):
            train_apollo_probe(features, metadata, self.root / "bad-validation-pair")
        metadata.write_text(original, encoding="utf-8")
        with metadata.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows.append({**rows[0], "example_id": "extra-honest"})
        duplicated_metadata = self.metadata("duplicate-condition.csv", rows)
        with np.load(features, allow_pickle=False) as archive:
            duplicated_features = self.features(
                "duplicate-condition.npz",
                np.vstack([archive["X"], [[-2]]]),
                [*archive["example_ids"].tolist(), "extra-honest"],
            )
        with self.assertRaisesRegex(ValueError, "Train pair.*exactly once"):
            train_apollo_probe(
                duplicated_features, duplicated_metadata, self.root / "bad-training-pair"
            )

    def test_metadata_exact_match_and_instruction_target_are_required(self):
        features, metadata = self.training()
        metadata.write_text(
            metadata.read_text(encoding="utf-8").replace("vd,val-pair", "x,val-pair"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "exact ID"):
            train_apollo_probe(features, metadata, self.root / "mismatch")

    def test_calibration_ties_and_disjoint_audited_evaluation(self):
        controls = self.scores("controls.csv", [("c1", 1), ("c2", 1), ("c3", 1)])
        threshold = self.root / "threshold.json"
        calibration = calibrate_control_threshold(controls, threshold)
        self.assertEqual(calibration["threshold"], 1)
        self.assertEqual(calibration["empirical_calibration_fpr"], 0)
        scores = self.scores("eval.csv", [("e1", 1), ("e2", 2)])
        metadata = self.metadata(
            "eval_labels.csv",
            [
                {
                    "example_id": identifier,
                    "pair_id": "held-out",
                    "split": "evaluation",
                    "label": label,
                    "label_target": "audited_commission_lie",
                    "label_audited": "true",
                }
                for identifier, label in (("e1", 0), ("e2", 1))
            ],
        )
        report = evaluate_apollo_scores(scores, metadata, self.root / "result.json", threshold)
        self.assertEqual(report["roc_auc"], 1)
        self.assertEqual(report["false_positive_rate"], 0)
        self.assertEqual(report["true_positive_rate"], 1)
        self.assertNotIn("brier_score", report)
        metadata.write_text(
            metadata.read_text(encoding="utf-8").replace("e1,", "c1,"), encoding="utf-8"
        )
        overlapping = self.scores("overlap.csv", [("c1", 1), ("e2", 2)])
        with self.assertRaisesRegex(ValueError, "overlap control"):
            evaluate_apollo_scores(overlapping, metadata, self.root / "overlap.json", threshold)

    def test_evaluation_requires_audit_and_rejects_instruction_labels(self):
        scores = self.scores("eval.csv", [("e1", 0), ("e2", 1)])
        rows = [
            {
                "example_id": identifier,
                "pair_id": "p",
                "split": "evaluation",
                "label": label,
                "label_target": "audited_behavior",
            }
            for identifier, label in (("e1", 0), ("e2", 1))
        ]
        metadata = self.metadata("eval.csv.labels", rows)
        with self.assertRaisesRegex(ValueError, "label_audited"):
            evaluate_apollo_scores(scores, metadata, self.root / "unaudited.json")
        for row in rows:
            row.update(label_target=INSTRUCTION_TARGET, label_audited="true")
        metadata = self.metadata("instruction.csv", rows)
        with self.assertRaisesRegex(ValueError, "not instruction"):
            evaluate_apollo_scores(scores, metadata, self.root / "instruction.json")

    def test_nonfinite_scores_duplicate_ids_and_training_controls_rejected(self):
        for name, values in (
            ("nan.csv", [("a", float("nan"))]),
            ("duplicate.csv", [("a", 0), ("a", 1)]),
        ):
            scores = self.scores(name, values)
            with self.assertRaises(ValueError):
                calibrate_control_threshold(scores, self.root / (name + ".json"))
        features, metadata = self.training()
        train_apollo_probe(features, metadata, self.root / "probe")
        score_apollo_probe(features, self.root / "probe", self.root / "train_scores.csv")
        with self.assertRaisesRegex(ValueError, "overlaps probe"):
            calibrate_control_threshold(self.root / "train_scores.csv", self.root / "bad.json")


if __name__ == "__main__":
    unittest.main()
