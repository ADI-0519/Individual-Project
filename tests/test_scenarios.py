import copy
import json
import tempfile
import unittest
from pathlib import Path

from latent_probes.schema import Scenario, ScenarioValidationError, load_scenarios

DATA = Path(__file__).resolve().parents[1] / "data" / "pilot_scenarios.jsonl"


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = load_scenarios(DATA)

    def setUp(self):
        self.record = self.scenarios[0].to_dict()

    def test_development_data_has_balanced_outcomes_and_group_labels(self):
        self.assertEqual(len(self.scenarios), 24)
        self.assertEqual(len({s.family_id for s in self.scenarios}), 6)
        self.assertEqual(len({s.domain for s in self.scenarios}), 3)
        self.assertEqual(sum(s.full_information_answer == "yes" for s in self.scenarios), 12)
        self.assertEqual(len({s.family_id for s in self.scenarios[:4]}), 4)

    def test_roundtrip_preserves_every_field(self):
        for scenario in self.scenarios:
            self.assertEqual(Scenario.from_dict(scenario.to_dict()), scenario)

    def test_controls_only_assert_source_facts(self):
        for scenario in self.scenarios:
            source = {fact.id: fact.text for fact in scenario.facts}
            self.assertEqual(
                scenario.control_messages["full_disclosure"],
                " ".join(fact.text for fact in scenario.facts),
            )
            self.assertEqual(
                scenario.control_messages["selective"],
                " ".join(source[key] for key in scenario.selective_fact_ids),
            )
            self.assertNotIn(scenario.critical_fact.text, scenario.control_messages["selective"])

    def test_rejects_duplicate_fact_ids(self):
        self.record["facts"].append(copy.deepcopy(self.record["facts"][0]))
        with self.assertRaises(ScenarioValidationError):
            Scenario.from_dict(self.record)

    def test_rejects_missing_critical_reference(self):
        self.record["critical_fact_id"] = "absent"
        with self.assertRaises(ScenarioValidationError):
            Scenario.from_dict(self.record)

    def test_rejects_critical_fact_in_selective_control(self):
        self.record["selective_fact_ids"].append(self.record["critical_fact_id"])
        with self.assertRaises(ScenarioValidationError):
            Scenario.from_dict(self.record)

    def test_rejects_missing_duplicate_or_empty_selected_references(self):
        for values in (["absent"], ["performance", "performance"], []):
            with self.subTest(values=values):
                record = copy.deepcopy(self.record)
                record["selective_fact_ids"] = values
                with self.assertRaises(ScenarioValidationError):
                    Scenario.from_dict(record)

    def test_rejects_unexpected_field_that_could_hide_labels(self):
        self.record["deception_label"] = True
        with self.assertRaises(ScenarioValidationError):
            Scenario.from_dict(self.record)

    def test_rejects_invalid_answers_and_nonstring_facts(self):
        for answer in ("unknown", "YES", [], True):
            with self.subTest(answer=answer):
                record = copy.deepcopy(self.record)
                record["full_information_answer"] = answer
                with self.assertRaises(ScenarioValidationError):
                    Scenario.from_dict(record)
        self.record["facts"][0]["text"] = 123
        with self.assertRaises(ScenarioValidationError):
            Scenario.from_dict(self.record)

    def test_loader_reports_bad_line_and_duplicate_scenario(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.jsonl"
            path.write_text(json.dumps(self.record) + "\nnot json\n", encoding="utf-8")
            with self.assertRaisesRegex(ScenarioValidationError, r":2:"):
                load_scenarios(path)
            path.write_text((json.dumps(self.record) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ScenarioValidationError, "duplicate scenario_id"):
                load_scenarios(path)

    def test_loader_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "empty.jsonl"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ScenarioValidationError, "no scenarios"):
                load_scenarios(path)

    def test_irrelevant_omission_control_is_retained(self):
        control = next(s for s in self.scenarios if s.scenario_id == "backup_grove")
        self.assertEqual(control.full_information_answer, "yes")
        self.assertIn("Backup A for Grove passed", control.control_messages["selective"])
        self.assertIn("harmless-omission", control.notes)


if __name__ == "__main__":
    unittest.main()
