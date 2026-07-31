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
        # New fields
        self.assertIn("valid", got)
        self.assertIn("invalid_reasons", got)
        self.assertIn("finish_reason", got)

    def test_extract_acceptance_is_null_for_base(self):
        got = MOD.extract_measurement({
            "content": "ok",
            "tokens_predicted": 10,
            "timings": {"predicted_per_second": 20.0},
        })
        self.assertIsNone(got["draft_accept_pct"])

    def test_extract_valid_fields_present(self):
        """Test that new validity fields are present."""
        got = MOD.extract_measurement({
            "choices": [{"text": "some valid code", "finish_reason": "stop"}],
            "usage": {"completion_tokens": 20},
            "timings": {"predicted_per_second": 20.0},
        })
        self.assertIn("valid", got)
        self.assertIn("invalid_reasons", got)
        self.assertIn("finish_reason", got)
        self.assertIn("draft_n", got)
        self.assertIn("draft_n_accepted", got)

    def test_summary_stats(self):
        result = MOD.summary_stats([10.0, 11.0, 30.0])
        self.assertEqual(result["median"], 11.0)
        self.assertEqual(result["min"], 10.0)
        self.assertEqual(result["max"], 30.0)
        self.assertEqual(result["mad"], 1.0)
        self.assertEqual(result["count"], 3)


class TestExtractPythonCode(unittest.TestCase):
    """Tests for Python code extraction."""

    def test_extract_fenced_python(self):
        content = "```python\ndef fibonacci(n):\n    return n\n```"
        code = MOD._extract_python_code(content)
        self.assertIn("def fibonacci", code)

    def test_extract_generic_fence(self):
        content = "```\ndef fibonacci(n):\n    return n\n```"
        code = MOD._extract_python_code(content)
        self.assertIn("def fibonacci", code)

    def test_no_fence_returns_none(self):
        content = "Just some text without code blocks"
        code = MOD._extract_python_code(content)
        self.assertIsNone(code)


if __name__ == "__main__":
    unittest.main()
