"""Feature/provenance and independent-import tests; all model outputs are fixtures."""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from latent_probes.backends import GenerationResult
from latent_probes.config import ExperimentConfig, GenerationSettings, ModelSettings
from latent_probes.features import pack_features
from latent_probes.import_examples import extract_examples, load_examples
from latent_probes.probes import split_groups
from latent_probes.storage import atomic_json, atomic_npz, canonical_hash, file_sha256, read_json


class FeaturePackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="feature-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.backend = {
            "backend": "explicit_test_fixture",
            "is_test_fixture": True,
            "model_id": "test-only",
            "resolved_revision": "fixture-revision",
            "dtype": "float32",
            "quantization": "none",
            "chat_template_sha256": "fixture-template",
            "extraction": "scripted_unit_vectors",
            "attention_implementation": "fixture",
            "positions": {"prompt_last": "fixture prompt", "response_last": "fixture response"},
        }
        # File ordering differs from ID ordering to catch accidental positional joins.
        for filename, identifier, offset, objective in (
            ("a_record", "z_example", 1, "neutral"),
            ("z_record", "a_example", 10, "favourable"),
        ):
            feature_path = self.run / "features" / f"{identifier}.npz"
            atomic_npz(
                feature_path,
                layer_0_prompt_last=np.asarray([offset, offset + 1], dtype=np.float32),
                layer_0_response_last=np.asarray([offset + 2, offset + 3], dtype=np.float32),
            )
            row = {
                "example_id": identifier,
                "scenario_id": f"scenario_{identifier}",
                "family_id": f"family_{identifier}",
                "objective": objective,
                "assumed_knowledge": False,
                "messages": [{"role": "user", "content": f"Fixture prompt {identifier}."}],
                "generation": {"text": "FIXTURE_ANSWER_TOKEN"},
                "feature_file": feature_path.relative_to(self.run).as_posix(),
                "feature_sha256": file_sha256(feature_path),
            }
            atomic_json(self.run / "sender" / f"{filename}.json", row)
        self.refresh_manifest()

    def refresh_manifest(self):
        rows = [read_json(path) for path in sorted((self.run / "sender").glob("*.json"))]
        atomic_json(
            self.run / "manifest.json",
            {
                "stage": "sender",
                "status": "complete",
                "sender_count": len(rows),
                "sender_records_hash": canonical_hash(rows),
                "backend": self.backend,
                "identity": {"fixture": "not_research"},
            },
        )

    def annotations(self, identifiers=("a_example", "z_example"), label_target=None):
        path = self.root / "annotations.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            fields = ["example_id", "label", "critical_fact_disclosed"]
            if label_target is not None:
                fields.append("label_target")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, identifier in enumerate(identifiers):
                row = {
                    "example_id": identifier,
                    "label": index % 2,
                    "critical_fact_disclosed": 1 - index % 2,
                }
                if label_target is not None:
                    row["label_target"] = label_target
                writer.writerow(row)
        return path

    def test_pack_preserves_vector_id_alignment_and_declared_capture(self):
        output = self.root / "packed"
        summary = pack_features(self.run, output, 0, "response_last")
        with np.load(output / "features.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["example_ids"].tolist(), ["z_example", "a_example"])
            np.testing.assert_array_equal(arrays["X"], [[3, 4], [12, 13]])
            signature = json.loads(str(arrays["feature_signature"].item()))
        self.assertTrue(signature["is_test_fixture"])
        self.assertEqual(signature["position"], "response_last")
        self.assertEqual(signature["decoder_block_index"], 0)
        self.assertEqual(summary["dimension"], 2)
        with (output / "metadata.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["example_id"] for row in rows], ["z_example", "a_example"])
        self.assertTrue(all(row["label"] == "" for row in rows))
        self.assertEqual([row["objective_pressure"] for row in rows], ["0", "1"])
        self.assertIn("FIXTURE_ANSWER_TOKEN", rows[0]["text"])

    def test_prompt_last_text_baseline_excludes_future_answer(self):
        output = self.root / "packed"
        pack_features(self.run, output, 0, "prompt_last")
        with (output / "metadata.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertNotIn("FIXTURE_ANSWER_TOKEN", rows[0]["text"])
        self.assertEqual(rows[0]["text"], rows[0]["prompt_text"])
        self.assertEqual(rows[0]["output_text"], "FIXTURE_ANSWER_TOKEN")

    def test_capture_signature_is_portable_across_run_directories(self):
        first = self.root / "first"
        pack_features(self.run, first, 0, "prompt_last")
        copied = self.root / "copied_run"
        shutil.copytree(self.run, copied)
        second = self.root / "second"
        pack_features(copied, second, 0, "prompt_last")
        with (
            np.load(first / "features.npz", allow_pickle=False) as left,
            np.load(second / "features.npz", allow_pickle=False) as right,
        ):
            self.assertEqual(left["feature_signature"].item(), right["feature_signature"].item())

    def test_annotations_are_joined_by_id_not_row_position(self):
        output = self.root / "packed"
        pack_features(self.run, output, 0, annotations_path=self.annotations())
        with (output / "metadata.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            [(row["example_id"], row["label"]) for row in rows],
            [("z_example", "1"), ("a_example", "0")],
        )

    def test_manual_label_meaning_is_preserved_in_packed_metadata(self):
        output = self.root / "packed"
        annotations = self.annotations(label_target="audited_literal_false_statement")
        pack_features(self.run, output, 0, annotations_path=annotations)
        with (output / "metadata.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["label_target"] for row in rows}, {"audited_literal_false_statement"})

    def test_annotation_ids_must_match_exactly(self):
        for identifiers in (("a_example",), ("a_example", "z_example", "extra")):
            with self.subTest(identifiers=identifiers):
                with self.assertRaisesRegex(ValueError, "match.*exactly"):
                    pack_features(
                        self.run,
                        self.root / "packed",
                        0,
                        annotations_path=self.annotations(identifiers),
                    )
                self.assertFalse((self.root / "packed").exists())

    def test_duplicate_annotation_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicated"):
            pack_features(
                self.run,
                self.root / "packed",
                0,
                annotations_path=self.annotations(("a_example", "a_example")),
            )

    def test_tampered_feature_bytes_are_rejected(self):
        path = self.run / "features" / "z_example.npz"
        atomic_npz(path, layer_0_response_last=np.asarray([999, 999]))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            pack_features(self.run, self.root / "packed", 0)

    def test_tampered_record_is_rejected(self):
        path = self.run / "sender" / "a_record.json"
        row = read_json(path)
        row["generation"]["text"] = "changed"
        atomic_json(path, row)
        with self.assertRaisesRegex(ValueError, "records changed"):
            pack_features(self.run, self.root / "packed", 0)

    def test_uncaptured_key_and_incomplete_runs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "was not captured"):
            pack_features(self.run, self.root / "packed", 3)
        path = self.run / "manifest.json"
        manifest = read_json(path)
        manifest["status"] = "failed"
        atomic_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "completed"):
            pack_features(self.run, self.root / "packed", 0)

    def test_feature_path_cannot_escape_run(self):
        outside = self.root / "outside.npz"
        atomic_npz(outside, layer_0_response_last=np.asarray([0, 1]))
        path = self.run / "sender" / "a_record.json"
        row = read_json(path)
        row["feature_file"] = "../outside.npz"
        row["feature_sha256"] = file_sha256(outside)
        atomic_json(path, row)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "escapes"):
            pack_features(self.run, self.root / "packed", 0)


class IndependentImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="independent-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "examples.jsonl"
        template = Path(__file__).resolve().parents[1] / "data" / "probe_examples_template.jsonl"
        self.rows = [json.loads(line) for line in template.read_text().splitlines()]
        self.write(self.rows)

    def write(self, rows):
        self.path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def test_template_has_declared_balanced_targets_but_insufficient_families(self):
        rows = load_examples(self.path)
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["label"] for row in rows], [1, 0, 0, 1])
        self.assertTrue(all(row["label_target"] == "recipient_prior_belief_true" for row in rows))
        with self.assertRaisesRegex(ValueError, "three independent groups"):
            split_groups([row["family_id"] for row in rows])

    def test_duplicate_examples_and_external_responses_are_rejected(self):
        self.write([self.rows[0], self.rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_examples(self.path)
        self.write([{**self.rows[0], "response": "YES"}])
        with self.assertRaisesRegex(ValueError, "off-policy"):
            load_examples(self.path)

    def test_labels_are_strict_binary_integers(self):
        for value in (True, "1", 1.0, 2, -1):
            with self.subTest(value=value):
                self.write([{**self.rows[0], "label": value}])
                with self.assertRaisesRegex(ValueError, "integer 0/1"):
                    load_examples(self.path)

    def test_labels_require_nonempty_string_target(self):
        for value in ("", "  ", ["belief"], {"name": "belief"}):
            with self.subTest(value=value):
                self.write([{**self.rows[0], "label_target": value}])
                with self.assertRaises(ValueError):
                    load_examples(self.path)

    def test_unlabelled_generation_examples_are_allowed(self):
        row = dict(self.rows[0])
        row.pop("label")
        row.pop("label_target")
        self.write([row])
        self.assertNotIn("label", load_examples(self.path)[0])

    def test_malformed_objects_and_messages_have_clear_errors(self):
        cases = [
            [],
            {**self.rows[0], "messages": [1]},
            {**self.rows[0], "messages": [{"role": "assistant", "content": "YES"}]},
        ]
        for row in cases:
            with self.subTest(row=row):
                self.write([row])
                with self.assertRaises(ValueError):
                    load_examples(self.path)

    def test_import_preserves_supervision_but_never_prompts_with_labels(self):
        events = []
        metadata = {
            "backend": "explicit_test_fixture",
            "is_test_fixture": True,
            "model_id": "fixture-only",
            "resolved_revision": "fixture-revision",
            "dtype": "float32",
            "quantization": "none",
            "capture_layers": [0],
            "chat_template_sha256": "fixture",
            "extraction": "fixture",
            "positions": {"prompt_last": "fixture", "response_last": "fixture"},
        }

        class FixtureBackend:
            def __init__(self, settings):
                self.metadata = metadata

            def generate(self, messages, settings, seed, capture=False):
                events.append(messages)
                return GenerationResult(
                    text="YES",
                    features={
                        "layer_0_prompt_last": np.array([1, 2], dtype=np.float32),
                        "layer_0_response_last": np.array([3, 4], dtype=np.float32),
                    },
                )

            def close(self):
                pass

        config = ExperimentConfig(
            model=ModelSettings(
                model_id="fixture-only", device="cpu", dtype="float32", capture_layers=[0]
            ),
            sender=GenerationSettings(do_sample=False),
            seeds=[17],
        )
        output = self.root / "run"
        result = extract_examples(config, self.path, output, backend_factory=FixtureBackend)
        self.assertTrue(result["is_test_fixture"])
        self.assertEqual(len(events), 4)
        self.assertEqual(events, [row["messages"] for row in self.rows])
        records = [read_json(path) for path in sorted((output / "sender").glob("*.json"))]
        expected = {row["example_id"]: row for row in self.rows}
        for record in records:
            source = expected[record["source_example_id"]]
            self.assertEqual(record["supervision"]["label"], source["label"])
            self.assertEqual(record["supervision"]["notes"], source["notes"])
            self.assertEqual(record["generation"]["text"], "YES")
        # Deliberately wrong fixture answers must not overwrite ground-truth task labels.
        self.assertEqual(sum(row["supervision"]["label"] == 0 for row in records), 2)
        packed = self.root / "packed"
        pack_features(output, packed, 0, "prompt_last")
        with (packed / "metadata.csv").open() as handle:
            packed_rows = list(csv.DictReader(handle))
        self.assertEqual(sorted(row["label"] for row in packed_rows), ["0", "0", "1", "1"])
        # A valid resume skips work; an edited completed record is never blessed.
        extract_examples(config, self.path, output, resume=True, backend_factory=FixtureBackend)
        self.assertEqual(len(events), 4)
        changed_path = next((output / "sender").glob("*.json"))
        changed = read_json(changed_path)
        changed["generation"]["text"] = "NO"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "records changed"):
            extract_examples(config, self.path, output, resume=True, backend_factory=FixtureBackend)


if __name__ == "__main__":
    unittest.main()
