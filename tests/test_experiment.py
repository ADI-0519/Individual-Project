"""Pipeline integrity tests using explicit scripted fixtures, never model evidence.

Every generated record in these tests is stored in a temporary directory and
marked is_test_fixture=True. The vectors and answers are hand-programmed to
exercise software behaviour, not to simulate the research hypothesis.
"""

import csv
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from latent_probes.backends import GenerationResult
from latent_probes.config import ExperimentConfig, ModelSettings
from latent_probes.experiment import _load_backend, plan, replay_recipients, run_senders
from latent_probes.storage import read_json


class ScriptedBackend:
    """Explicit test fixture; no model loading or real inference occurs."""

    def __init__(self, settings, events, feature_directory=None):
        self.events = events
        self.feature_directory = feature_directory
        self.closed = False
        self.metadata = {
            "backend": "scripted_unit_test_fixture",
            "is_test_fixture": True,
            "model_id": settings.model_id,
            "resolved_revision": "fixture-revision",
            "dtype": settings.dtype,
            "quantization": settings.quantization,
            "chat_template_sha256": "fixture-template",
            "capture_layers": settings.capture_layers,
            "extraction": "scripted_unit_test_vectors",
            "positions": {"prompt_last": "fixture", "response_last": "fixture"},
        }

    def generate(self, messages, settings, seed, capture=False):
        is_forecast = messages[-1]["content"].startswith("Diagnostic forecast:")
        event = {
            "capture": capture,
            "is_forecast": is_forecast,
            "messages": [dict(message) for message in messages],
            "seed": seed,
        }
        if is_forecast and self.feature_directory is not None:
            event["saved_feature_count"] = len(list(self.feature_directory.glob("*.npz")))
        self.events.append(event)
        if capture:
            return GenerationResult(
                text="SCRIPTED UNIT TEST MESSAGE; this is not a model finding.",
                prompt_token_ids=[101, 102],
                generated_token_ids=[201, 202],
                features={
                    "layer_0_prompt_last": np.arange(4, dtype=np.float32),
                    "layer_0_response_last": np.arange(4, dtype=np.float32) + 1,
                },
                token_positions={"prompt_last": 1, "response_last": 3},
                finish_reason="eos",
            )
        # Exercise all recipient outcome classes, including malformed output.
        text = (
            "UNKNOWN"
            if is_forecast
            else ("YES", "NO", "UNKNOWN", "YES or NO")[len(self.events) % 4]
        )
        return GenerationResult(
            text=text,
            prompt_token_ids=[301],
            generated_token_ids=[401],
            finish_reason="eos",
        )

    def close(self):
        self.closed = True


class ExperimentTests(unittest.TestCase):
    def test_resume_refuses_changed_attention_implementation(self):
        settings = ModelSettings(model_id="fixture", capture_layers=[0])
        backend = ScriptedBackend(settings, [])
        previous = dict(backend.metadata, attention_implementation="sdpa")
        backend.metadata["attention_implementation"] = "eager"
        with self.assertRaisesRegex(ValueError, "differs"):
            _load_backend(settings, {"backend": previous}, lambda _: backend)
        self.assertTrue(backend.closed)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="latent-probes-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        dataset = Path(__file__).resolve().parents[1] / "data" / "pilot_scenarios.jsonl"
        self.scenarios_path = self.root / "two_scenarios.jsonl"
        self.scenarios_path.write_text(
            "\n".join(dataset.read_text(encoding="utf-8").splitlines()[:2]) + "\n",
            encoding="utf-8",
        )
        self.output = self.root / "sender_run"
        self.config = ExperimentConfig(
            model=ModelSettings(model_id="unit-test-fixture", capture_layers=[0]),
            seeds=[17],
        )
        self.events = []
        self.backends = []
        # Isolate provenance tests from concurrent edits elsewhere in this
        # repository while preserving the actual resume comparisons.
        self.code_patch = patch(
            "latent_probes.experiment.code_fingerprint", return_value="fixture-code"
        )
        self.environment_patch = patch(
            "latent_probes.experiment.environment_info",
            return_value={
                "python": "unit-test-fixture",
                "platform": "unit-test-fixture",
                "packages": {"fixture": "1"},
                "git_commit": None,
                "code_sha256": "fixture-code",
            },
        )
        self.code_patch.start()
        self.environment_patch.start()
        self.addCleanup(self.code_patch.stop)
        self.addCleanup(self.environment_patch.stop)

    def factory(self, settings):
        backend = ScriptedBackend(settings, self.events, self.output / "features")
        self.backends.append(backend)
        return backend

    def run_fixture(self, config=None):
        return run_senders(
            config or self.config, self.scenarios_path, self.output, backend_factory=self.factory
        )

    def sender_rows(self):
        return [read_json(path) for path in sorted((self.output / "sender").glob("*.json"))]

    def test_plan_has_counts_and_no_model_outputs(self):
        output = self.root / "plan.json"
        result = plan(self.config, self.scenarios_path, output)
        self.assertEqual(result["sender_generations"], 8)
        self.assertEqual(result["recipient_generations"], 28)
        self.assertEqual(result["status"], "plan_only_no_model_outputs")
        payload = read_json(output)
        self.assertEqual(len(payload["control_recipient_examples"]), 12)
        self.assertTrue(all("generation" not in job for job in payload["sender_jobs"]))
        self.assertEqual(self.events, [])
        with self.assertRaises(FileExistsError):
            plan(self.config, self.scenarios_path, output)

    def test_senders_save_exact_provenance_and_blank_annotations(self):
        result = self.run_fixture()
        self.assertEqual(result["sender_count"], 8)
        self.assertTrue(result["is_test_fixture"])
        self.assertEqual(len(self.events), 8)
        rows = self.sender_rows()
        combinations = {
            (row["scenario_id"], row["objective"], row["assumed_knowledge"], row["seed"])
            for row in rows
        }
        self.assertEqual(len(combinations), 8)
        for row in rows:
            self.assertNotIn("actual_knowledge", row)
            self.assertTrue(row["is_test_fixture"])
            self.assertEqual(row["generation"]["generated_token_ids"], [201, 202])
            self.assertEqual(row["generation"]["token_positions"]["response_last"], 3)
            with np.load(self.output / row["feature_file"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["layer_0_response_last"], [1, 2, 3, 4])
        with (self.output / "annotations.csv").open(newline="", encoding="utf-8") as handle:
            annotations = list(csv.DictReader(handle))
        self.assertEqual(len(annotations), 8)
        self.assertTrue(
            all(
                row["label"] == ""
                and row["label_target"] == ""
                and row["critical_fact_disclosed"] == ""
                for row in annotations
            )
        )
        self.assertTrue(all(backend.closed for backend in self.backends))

    def test_recipient_replay_crosses_actual_knowledge_and_keeps_controls(self):
        self.run_fixture()
        self.events.clear()
        result = replay_recipients(self.output, backend_factory=self.factory)
        self.assertEqual(result["recipient_count"], 28)
        self.assertTrue(result["is_test_fixture"])
        rows = [
            read_json(path) for path in sorted((Path(result["output"]) / "records").glob("*.json"))
        ]
        kinds = Counter(row["message_kind"] for row in rows)
        self.assertEqual(
            kinds, {"generated": 16, "no_message": 4, "full_disclosure": 4, "selective_control": 4}
        )
        by_sender = defaultdict(list)
        for row in rows:
            self.assertTrue(row["is_test_fixture"])
            joined_prompt = "\n".join(message["content"] for message in row["messages"])
            self.assertNotIn("Your objective", joined_prompt)
            self.assertNotIn("Complete source available to you", joined_prompt)
            if row["sender_example_id"]:
                by_sender[row["sender_example_id"]].append(row)
        self.assertEqual(len(by_sender), 8)
        for pair in by_sender.values():
            self.assertEqual({row["actual_knowledge"] for row in pair}, {False, True})
            self.assertEqual(pair[0]["message"], pair[1]["message"])
        self.assertEqual({row["answer"] for row in rows}, {"yes", "no", "unknown", "invalid"})
        self.assertTrue(all(not event["capture"] for event in self.events))
        self.assertTrue(all(backend.closed for backend in self.backends))

    def test_forecasts_follow_saved_passive_capture(self):
        config = replace(self.config, elicited_forecast=True)
        self.run_fixture(config)
        self.assertEqual(len(self.events), 16)
        for index in range(0, len(self.events), 2):
            capture, forecast = self.events[index : index + 2]
            self.assertTrue(capture["capture"])
            self.assertFalse(capture["is_forecast"])
            self.assertFalse(forecast["capture"])
            self.assertTrue(forecast["is_forecast"])
            self.assertEqual(forecast["saved_feature_count"], index // 2 + 1)
            self.assertNotIn("Diagnostic forecast:", str(capture["messages"]))
        self.assertTrue(
            all(row["elicited_forecast"]["answer"] == "unknown" for row in self.sender_rows())
        )

    def test_existing_run_requires_explicit_resume_and_skips_completed_work(self):
        self.run_fixture()
        calls_before = len(self.events)
        with self.assertRaises(FileExistsError):
            self.run_fixture()
        result = run_senders(
            self.config, self.scenarios_path, self.output, resume=True, backend_factory=self.factory
        )
        self.assertEqual(result["sender_count"], 8)
        self.assertEqual(len(self.events), calls_before)
        replay_recipients(self.output, backend_factory=self.factory)
        replay_calls = len(self.events)
        with self.assertRaises(FileExistsError):
            replay_recipients(self.output, backend_factory=self.factory)
        result = replay_recipients(self.output, resume=True, backend_factory=self.factory)
        self.assertEqual(result["recipient_count"], 28)
        self.assertEqual(len(self.events), replay_calls)

    def test_resume_rejects_configuration_and_code_changes(self):
        self.run_fixture()
        changed = replace(self.config, seeds=[29])
        with self.assertRaisesRegex(ValueError, "Resume inputs differ"):
            run_senders(
                changed, self.scenarios_path, self.output, resume=True, backend_factory=self.factory
            )
        with patch("latent_probes.experiment.code_fingerprint", return_value="changed-code"):
            with self.assertRaisesRegex(ValueError, "Source code changed"):
                run_senders(
                    self.config,
                    self.scenarios_path,
                    self.output,
                    resume=True,
                    backend_factory=self.factory,
                )

    def test_resume_rejects_record_without_its_activation_file(self):
        self.run_fixture()
        first = self.sender_rows()[0]
        (self.output / first["feature_file"]).unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            run_senders(
                self.config,
                self.scenarios_path,
                self.output,
                resume=True,
                backend_factory=self.factory,
            )
        self.assertEqual(read_json(self.output / "manifest.json")["status"], "failed")
        self.assertTrue(self.backends[-1].closed)

    def test_replay_rejects_changed_source_snapshot(self):
        self.run_fixture()
        snapshot_path = self.output / "scenarios.jsonl"
        rows = [json.loads(line) for line in snapshot_path.read_text(encoding="utf-8").splitlines()]
        rows[0]["facts"][0]["text"] += " Changed after generation."
        snapshot_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            replay_recipients(self.output, backend_factory=self.factory)

    def test_replay_rejects_changed_saved_config_even_with_override(self):
        self.run_fixture()
        config_path = self.output / "config.json"
        saved = read_json(config_path)
        saved["seeds"] = [999]
        config_path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Saved sender configuration"):
            replay_recipients(self.output, config=self.config, backend_factory=self.factory)

    def test_resume_rejects_changed_activation_bytes(self):
        self.run_fixture()
        first = self.sender_rows()[0]
        feature_path = self.output / first["feature_file"]
        with np.load(feature_path, allow_pickle=False) as archive:
            changed = {key: archive[key].copy() for key in archive.files}
        changed["layer_0_response_last"][0] += 100
        np.savez_compressed(feature_path, **changed)
        with self.assertRaisesRegex(ValueError, "activation checksum"):
            run_senders(
                self.config,
                self.scenarios_path,
                self.output,
                resume=True,
                backend_factory=self.factory,
            )

    def test_extra_sender_record_is_rejected(self):
        self.run_fixture()
        extra = self.output / "sender" / "unexpected.json"
        extra.write_text(json.dumps(self.sender_rows()[0]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unexpected sender record"):
            replay_recipients(self.output, backend_factory=self.factory)
        with self.assertRaisesRegex(ValueError, "Unexpected sender record"):
            run_senders(
                self.config,
                self.scenarios_path,
                self.output,
                resume=True,
                backend_factory=self.factory,
            )

    def test_recipient_resume_rejects_record_config_and_extra_file_changes(self):
        self.run_fixture()
        result = replay_recipients(self.output, backend_factory=self.factory)
        recipient_dir = Path(result["output"])
        manifest_path = recipient_dir / "manifest.json"
        original_manifest = read_json(manifest_path)
        first_path = sorted((recipient_dir / "records").glob("*.json"))[0]
        original_record = read_json(first_path)
        altered_record = dict(original_record)
        altered_record["answer"] = "no" if original_record["answer"] != "no" else "yes"
        first_path.write_text(json.dumps(altered_record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "recipient records differ"):
            replay_recipients(self.output, resume=True, backend_factory=self.factory)
        first_path.write_text(json.dumps(original_record), encoding="utf-8")
        manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")

        config_path = recipient_dir / "config.json"
        original_config = read_json(config_path)
        altered_config = dict(original_config)
        altered_config["seeds"] = [999]
        config_path.write_text(json.dumps(altered_config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Saved recipient configuration"):
            replay_recipients(self.output, resume=True, backend_factory=self.factory)
        config_path.write_text(json.dumps(original_config), encoding="utf-8")
        manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")

        (recipient_dir / "records" / "unexpected.json").write_text(
            json.dumps(original_record), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Unexpected recipient record"):
            replay_recipients(self.output, resume=True, backend_factory=self.factory)


if __name__ == "__main__":
    unittest.main()
