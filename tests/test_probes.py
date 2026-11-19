"""Synthetic numerical fixtures test the implementation, not the research hypothesis."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from latent_probes.probes import predict_probe, split_groups, train_probe


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(7)
        self.y = np.tile([0, 1, 0, 1], 10)
        self.groups = np.repeat([f"family-{index}" for index in range(10)], 4)
        self.ids = np.asarray([f"test-fixture-{index}" for index in range(len(self.y))])
        self.X = rng.normal(size=(len(self.y), 5))
        self.X[:, 0] += self.y * 4
        self.features = self.root / "features.npz"
        self.labels = self.root / "labels.csv"
        self.write_features()
        self.write_labels()

    def tearDown(self):
        self.tmp.cleanup()

    def write_features(self, signature="fixture-only", X=None, ids=None):
        values = {"X": self.X if X is None else X, "example_ids": self.ids if ids is None else ids}
        if signature is not None:
            values["feature_signature"] = np.asarray(signature)
        np.savez_compressed(self.features, **values)

    def write_labels(self, labels=None, text=False, scenario_groups=False, label_targets=None):
        fields = ["example_id", "family_id", "label"]
        if label_targets is not None:
            fields.append("label_target")
        if text:
            fields.append("text")
        if scenario_groups:
            fields.append("scenario_id")
        with self.labels.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(len(self.ids)):
                row = {
                    "example_id": self.ids[index],
                    "family_id": self.groups[index],
                    "label": self.y[index] if labels is None else labels[index],
                }
                if label_targets is not None:
                    row["label_target"] = label_targets[index]
                if text:
                    row["text"] = (
                        "accepted factual complete"
                        if self.y[index]
                        else "rejected factual selective"
                    )
                if scenario_groups:
                    row["scenario_id"] = f"scenario-{index}"
                writer.writerow(row)

    def test_group_split_is_disjoint_and_repeatable(self):
        first, second = split_groups(self.groups), split_groups(self.groups)
        seen = set()
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            selected = set(self.groups[first[name]])
            self.assertFalse(seen & selected)
            seen |= selected
        self.assertEqual(seen, set(self.groups))

    def test_split_is_independent_of_row_order(self):
        original = split_groups(self.groups)
        reordered = self.groups[::-1]
        changed = split_groups(reordered)
        for name in original:
            self.assertEqual(set(self.groups[original[name]]), set(reordered[changed[name]]))

    def test_roundtrip_predictions_and_grouped_metrics(self):
        output = self.root / "probe"
        manifest = train_probe(
            self.features, self.labels, output, C_grid=(0.1, 1.0), bootstrap_samples=40
        )
        self.assertTrue(manifest["capture_signature_present"])
        self.assertFalse(manifest["feature_provenance_verified"])
        self.assertEqual(manifest["bootstrap"]["n_groups"], 2)
        self.assertEqual(manifest["bootstrap"]["valid_draws"], 40)
        self.assertEqual(manifest["threshold"], 0.5)
        predictions = self.root / "predictions.csv"
        predict_probe(self.features, output, predictions)
        with predictions.open() as handle:
            all_scores = {row["example_id"]: float(row["score"]) for row in csv.DictReader(handle)}
        with (output / "test_predictions.csv").open() as handle:
            for row in csv.DictReader(handle):
                self.assertAlmostEqual(
                    all_scores[row["example_id"]], float(row["score"]), places=12
                )
        json.loads((output / "manifest.json").read_text())
        with np.load(output / "probe.npz", allow_pickle=False) as saved:
            train_ids = set(manifest["splits"]["train"]["example_ids"])
            train_rows = [
                index for index, identifier in enumerate(self.ids) if identifier in train_ids
            ]
            np.testing.assert_allclose(saved["mean"], self.X[train_rows].mean(axis=0))

    def test_text_baseline_uses_same_test_set(self):
        self.write_labels(text=True)
        manifest = train_probe(
            self.features, self.labels, self.root / "probe", C_grid=(0.1,), bootstrap_samples=0
        )
        self.assertTrue(manifest["text_baseline"]["enabled"])
        self.assertEqual(
            manifest["test"]["n_examples"], manifest["text_baseline"]["test"]["n_examples"]
        )
        self.assertFalse(manifest["text_baseline"]["information_parity_verified"])

    def test_rejects_duplicate_feature_ids(self):
        duplicate = self.ids.copy()
        duplicate[-1] = duplicate[0]
        self.write_features(ids=duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            train_probe(self.features, self.labels, self.root / "probe")

    def test_rejects_unlabelled_examples(self):
        labels = self.y.astype(str)
        labels[3] = ""
        self.write_labels(labels=labels)
        with self.assertRaisesRegex(ValueError, "explicitly 0 or 1"):
            train_probe(self.features, self.labels, self.root / "probe")

    def test_rejects_mixed_label_meanings_even_with_identical_binary_labels(self):
        targets = ["recipient_believes_proposition"] * len(self.ids)
        # Both rows carry label 0, but a zero has a different meaning in row 2.
        targets[2] = "source_proposition_is_true"
        self.write_labels(label_targets=targets)
        with self.assertRaisesRegex(ValueError, "multiple label_target meanings"):
            train_probe(self.features, self.labels, self.root / "probe")

    def test_rejects_missing_or_placeholder_declared_label_meaning(self):
        for missing in ("", "   ", "manual_definition_required"):
            with self.subTest(missing=missing):
                targets = ["recipient_believes_proposition"] * len(self.ids)
                targets[0] = missing
                self.write_labels(label_targets=targets)
                with self.assertRaisesRegex(ValueError, "label_target.*meaning"):
                    train_probe(self.features, self.labels, self.root / "probe")

    def test_records_consistent_label_meaning_without_claiming_validation(self):
        self.write_labels(label_targets=["recipient_believes_proposition"] * len(self.ids))
        manifest = train_probe(
            self.features, self.labels, self.root / "probe", C_grid=(0.1,), bootstrap_samples=0
        )
        semantics = manifest["target_semantics"]
        self.assertEqual(semantics["definition"], "recipient_believes_proposition")
        self.assertTrue(semantics["consistency_checked"])
        self.assertFalse(semantics["independently_validated"])

    def test_legacy_labels_keep_an_explicit_semantics_warning(self):
        manifest = train_probe(
            self.features, self.labels, self.root / "probe", C_grid=(0.1,), bootstrap_samples=0
        )
        self.assertIsNone(manifest["target_semantics"]["definition"])
        self.assertFalse(manifest["target_semantics"]["consistency_checked"])
        self.assertTrue(any("label_target declaration" in note for note in manifest["limitations"]))

    def test_named_instruction_target_does_not_inherit_generic_label_meanings(self):
        self.write_labels(
            label_targets=[f"unrelated-target-{index}" for index in range(len(self.ids))]
        )
        with self.labels.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with self.labels.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*rows[0], "objective_pressure"])
            writer.writeheader()
            for row, label in zip(rows, self.y, strict=True):
                writer.writerow({**row, "objective_pressure": int(label)})
        manifest = train_probe(
            self.features,
            self.labels,
            self.root / "probe",
            target="objective_pressure",
            C_grid=(0.1,),
            bootstrap_samples=0,
        )
        self.assertIn("instruction condition", manifest["target_semantics"]["definition"])
        self.assertIn("not deception", manifest["target_semantics"]["definition"])

    def test_rejects_missing_labels_instead_of_silent_join(self):
        lines = self.labels.read_text().splitlines()
        self.labels.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            train_probe(self.features, self.labels, self.root / "probe")

    def test_rejects_known_family_leakage(self):
        self.write_labels(scenario_groups=True)
        with self.assertRaisesRegex(ValueError, "spans multiple"):
            train_probe(self.features, self.labels, self.root / "probe", group_column="scenario_id")

    def test_rejects_single_class_partition(self):
        partitions = split_groups(self.groups)
        y = self.y.copy()
        y[partitions["test"]] = 0
        self.write_labels(labels=y)
        with self.assertRaisesRegex(ValueError, "preselected test partition"):
            train_probe(self.features, self.labels, self.root / "probe")

    def test_signature_prevents_cross_model_prediction(self):
        output = self.root / "probe"
        train_probe(self.features, self.labels, output, C_grid=(0.1,), bootstrap_samples=0)
        self.write_features(signature="different-model-same-width")
        with self.assertRaisesRegex(ValueError, "signature differs"):
            predict_probe(self.features, output, self.root / "bad.csv")

    def test_missing_signature_rejected_if_training_had_one(self):
        output = self.root / "probe"
        train_probe(self.features, self.labels, output, C_grid=(0.1,), bootstrap_samples=0)
        self.write_features(signature=None)
        with self.assertRaisesRegex(ValueError, "signature differs"):
            predict_probe(self.features, output, self.root / "bad.csv")

    def test_nonfinite_features_rejected(self):
        X = self.X.copy()
        X[1, 1] = np.nan
        self.write_features(X=X)
        with self.assertRaisesRegex(ValueError, "NaN"):
            train_probe(self.features, self.labels, self.root / "probe")

    def test_results_are_not_overwritten(self):
        output = self.root / "probe"
        train_probe(self.features, self.labels, output, C_grid=(0.1,), bootstrap_samples=0)
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            train_probe(self.features, self.labels, output)


if __name__ == "__main__":
    unittest.main()
