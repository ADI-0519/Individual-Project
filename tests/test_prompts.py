import inspect
import unittest
from pathlib import Path

from latent_probes.prompts import (
    build_forecast_messages,
    build_recipient_messages,
    build_sender_messages,
    parse_answer,
)
from latent_probes.schema import load_scenarios

DATA = Path(__file__).resolve().parents[1] / "data" / "pilot_scenarios.jsonl"


def text(messages):
    return "\n".join(message["content"] for message in messages)


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = load_scenarios(DATA)[0]

    def test_actual_reader_state_has_no_sender_input_parameter(self):
        self.assertEqual(
            set(inspect.signature(build_sender_messages).parameters),
            {"scenario", "objective", "assumed_knowledge"},
        )
        first = build_sender_messages(self.scenario, "neutral", False)
        for actual in (True, False):
            build_recipient_messages(self.scenario, "Nine tests passed.", actual)
            self.assertEqual(first, build_sender_messages(self.scenario, "neutral", False))

    def test_sender_receives_full_source_under_both_objectives(self):
        for objective in ("neutral", "favourable"):
            prompt = build_sender_messages(self.scenario, objective, False)
            for fact in self.scenario.facts:
                self.assertIn(fact.text, text(prompt))
            self.assertIn("Every factual assertion", prompt[0]["content"])

    def test_objective_changes_without_changing_factuality_requirement(self):
        neutral = build_sender_messages(self.scenario, "neutral", True)
        favourable = build_sender_messages(self.scenario, "favourable", True)
        self.assertEqual(neutral[0], favourable[0])
        self.assertNotEqual(neutral[1], favourable[1])

    def test_recipient_without_prior_receives_only_selective_facts(self):
        selective = self.scenario.control_messages["selective"]
        prompt = text(build_recipient_messages(self.scenario, selective, False))
        self.assertNotIn(self.scenario.critical_fact.text, prompt)
        self.assertIn(selective, prompt)
        self.assertNotIn("Your objective", prompt)
        self.assertNotIn("You are told", prompt)
        self.assertNotIn(self.scenario.notes, prompt)

    def test_recipient_prior_is_exactly_the_designated_fact(self):
        prompt = text(build_recipient_messages(self.scenario, None, True))
        self.assertIn(self.scenario.critical_fact.text, prompt)
        for fact in self.scenario.facts:
            if fact.id != self.scenario.critical_fact_id:
                self.assertNotIn(fact.text, prompt)

    def test_no_message_control_has_no_hidden_full_report(self):
        prompt = text(build_recipient_messages(self.scenario, None, False))
        self.assertIn("No report was supplied.", prompt)
        for fact in self.scenario.facts:
            self.assertNotIn(fact.text, prompt)

    def test_forecast_is_separate_continuation_and_does_not_mutate_sender(self):
        sender = build_sender_messages(self.scenario, "favourable", False)
        original = [dict(message) for message in sender]
        response = self.scenario.control_messages["selective"]
        forecast = build_forecast_messages(self.scenario, sender, response)
        self.assertEqual(sender, original)
        self.assertEqual(forecast[:-2], sender)
        self.assertEqual(forecast[-2], {"role": "assistant", "content": response})
        self.assertIn("Diagnostic forecast", forecast[-1]["content"])
        self.assertNotIn("actual_knowledge", inspect.signature(build_forecast_messages).parameters)

    def test_strict_answer_parser(self):
        valid = {" YES ": "yes", "no.": "no", "\nUNKNOWN\n": "unknown"}
        for value, expected in valid.items():
            self.assertEqual(parse_answer(value), expected)
        for value in (
            "",
            "YES or NO",
            "YES because all tests passed",
            '{"answer":"YES"}',
            "UNKNOWN!",
            None,
        ):
            with self.subTest(value=value):
                self.assertEqual(parse_answer(value), "invalid")

    def test_string_false_is_rejected_instead_of_becoming_truthy(self):
        with self.assertRaises(ValueError):
            build_sender_messages(self.scenario, "neutral", "false")
        with self.assertRaises(ValueError):
            build_recipient_messages(self.scenario, None, "false")

    def test_invalid_sender_objective_and_empty_report_are_rejected(self):
        with self.assertRaises(ValueError):
            build_sender_messages(self.scenario, "malformed", False)
        with self.assertRaises(ValueError):
            build_recipient_messages(self.scenario, " ", False)


if __name__ == "__main__":
    unittest.main()
