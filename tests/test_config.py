import unittest

from latent_probes.config import ExperimentConfig, GenerationSettings, ModelSettings


class ConfigurationTests(unittest.TestCase):
    def test_bad_numeric_sampling_values(self):
        for value in (float("nan"), float("inf"), True, "0.7", -0.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GenerationSettings(temperature=value)

    def test_no_duplicate_greedy_replicates(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(
                model=ModelSettings("test"),
                sender=GenerationSettings(do_sample=False),
                seeds=[1, 2],
            )

    def test_unknown_setting_rejected(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict({"model": {"model_id": "test", "max_input_token": 20}})

    def test_round_trip(self):
        config = ExperimentConfig(model=ModelSettings("test"))
        self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)


if __name__ == "__main__":
    unittest.main()
