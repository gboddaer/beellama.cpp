import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("vk_gap", HERE / "run.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class TestRunHarness(unittest.TestCase):

    def test_extract_timing_and_acceptance(self):
        response = {
            "choices": [{"text": "ok", "index": 0, "finish_reason": "length"}],
            "usage": {"completion_tokens": 200, "prompt_tokens": 50},
            "timings": {
                "predicted_per_second": 19.5,
                "draft_n": 100,
                "draft_n_accepted": 45,
            },
        }
        got = MOD.extract_measurement(response)
        self.assertEqual(got["tokens_predicted"], 200)
        self.assertEqual(got["predicted_per_second"], 19.5)
        self.assertEqual(got["draft_accept_pct"], 45.0)

    def test_extract_acceptance_is_null_for_base(self):
        got = MOD.extract_measurement({
            "content": "ok",
            "tokens_predicted": 10,
            "timings": {"predicted_per_second": 20.0},
        })
        self.assertIsNone(got["draft_accept_pct"])

    def test_summary_stats(self):
        result = MOD.summary_stats([10.0, 11.0, 30.0])
        self.assertEqual(result["median"], 11.0)
        self.assertEqual(result["min"], 10.0)
        self.assertEqual(result["max"], 30.0)
        self.assertEqual(result["mad"], 1.0)
