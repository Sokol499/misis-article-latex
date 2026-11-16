"""Run with: python3 -m unittest discover -s . -p 'test_*.py' -v"""
import unittest

from traffic_model import Scenario, raw_output_payloads


class TrafficModelTests(unittest.TestCase):
    def test_original_demo_example(self):
        self.assertEqual(raw_output_payloads(Scenario()), {
            "D0": 58_298_120,
            "D1": 4_020_560,
            "D2": 4_020_560,
            "D3": 160,
        })

    def test_one_generated_token(self):
        self.assertEqual(raw_output_payloads(Scenario(prompt_tokens=1, generated_tokens=1, vocab_size=10)), {
            "D0": 40, "D1": 40, "D2": 40, "D3": 8,
        })

    def test_batch_scaling(self):
        base = raw_output_payloads(Scenario())
        double = raw_output_payloads(Scenario(batch_size=2))
        for key in base:
            self.assertEqual(double[key], 2 * base[key])

    def test_d1_and_d2_same_communication(self):
        for l0 in (1, 8, 128):
            for steps in (1, 4, 20):
                v = raw_output_payloads(Scenario(prompt_tokens=l0, generated_tokens=steps))
                self.assertEqual(v["D1"], v["D2"])

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            Scenario(prompt_tokens=0)


if __name__ == "__main__":
    unittest.main()
