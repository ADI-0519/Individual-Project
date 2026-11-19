"""Summary checks on explicit hand-programmed fixtures, never model results."""

import tempfile
import unittest
from pathlib import Path

from latent_probes.config import ExperimentConfig, ModelSettings
from latent_probes.reporting import summarise_recipients
from latent_probes.storage import atomic_json, canonical_hash, read_json


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="latent-report-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "explicit_fixture"
        self.output = self.root / "summary"
        self.config = ExperimentConfig(
            model=ModelSettings(model_id="unit-test-fixture", capture_layers=[0]), seeds=[17]
        )
        atomic_json(self.source / "config.json", self.config.to_dict())
        self.rows = []
        for index, (answer, text) in enumerate(
            [("yes", "YES"), ("no", "NO"), ("unknown", "UNKNOWN"), ("invalid", "YES or NO")]
        ):
            row = {
                "example_id": f"r_fixture_{index}",
                "scenario_id": f"fixture_{index // 2}",
                "family_id": "fixture_family",
                "message_kind": "generated",
                "actual_knowledge": False,
                "objective": "neutral",
                "assumed_knowledge": False,
                "generation": {"text": text},
                "answer": answer,
                "full_information_answer": "no",
                "is_test_fixture": True,
            }
            row["record_sha256"] = canonical_hash(row)
            self.rows.append(row)
            atomic_json(self.source / "records" / f"{row['example_id']}.json", row)
        self.manifest = {
            "stage": "recipient",
            "status": "complete",
            "is_test_fixture": True,
            "recipient_count": 4,
            "recipient_records_hash": canonical_hash(self.rows),
            "identity": {"recipient_config_hash": canonical_hash(self.config.to_dict())},
        }
        atomic_json(self.source / "manifest.json", self.manifest)

    def test_summary_retains_unknown_invalid_and_all_run_denominator(self):
        result = summarise_recipients(self.source, self.output)
        self.assertTrue(result["is_test_fixture"])
        self.assertEqual(result["recipient_count"], 4)
        self.assertEqual(result["scenario_clusters"], 2)
        for counts in [result["overall"], result["groups"][0]]:
            self.assertEqual(counts["denominator_all_runs"], 4)
            self.assertEqual(counts["definite_answers"], 2)
            self.assertEqual(counts["valid_answers"], 3)
            self.assertEqual(counts["false_definite"], 1)
            self.assertEqual(counts["correct_definite"], 1)
            self.assertEqual(counts["unknown"], 1)
            self.assertEqual(counts["invalid"], 1)
            self.assertEqual(counts["false_definite_rate_all_runs"], 0.25)
            self.assertEqual(counts["unknown_rate_all_runs"], 0.25)
            self.assertEqual(counts["invalid_rate_all_runs"], 0.25)
        self.assertEqual(read_json(self.output / "summary.json"), result)
        self.assertTrue((self.output / "recipient_summary.csv").is_file())

    def test_summary_rejects_modified_record_before_writing(self):
        row = dict(self.rows[0])
        row["answer"] = "no"
        atomic_json(self.source / "records" / "r_fixture_0.json", row)
        with self.assertRaisesRegex(ValueError, "records differ"):
            summarise_recipients(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_summary_rejects_modified_configuration(self):
        saved = self.config.to_dict()
        saved["seeds"] = [29]
        atomic_json(self.source / "config.json", saved)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            summarise_recipients(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_summary_checks_answer_against_raw_generation(self):
        # Even a correctly checksummed file must have a correctly parsed answer.
        row = dict(self.rows[0])
        row["answer"] = "no"
        row.pop("record_sha256")
        row["record_sha256"] = canonical_hash(row)
        self.rows[0] = row
        atomic_json(self.source / "records" / "r_fixture_0.json", row)
        self.manifest["recipient_records_hash"] = canonical_hash(self.rows)
        atomic_json(self.source / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "answer differs"):
            summarise_recipients(self.source, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
