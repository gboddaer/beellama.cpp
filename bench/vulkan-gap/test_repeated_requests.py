"""Model-free tests for response validation, provenance, and persistent-server correctness."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("vk_gap", HERE / "run.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class TestResponseValidation(unittest.TestCase):
    """Tests for response classification — these fail until extract_measurement is updated."""

    def test_rejects_empty_one_token_sentinel(self):
        """Empty one-token response at implausible speed must be invalid."""
        response = {
            "choices": [{"text": "", "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
            "timings": {"predicted_per_second": 1_000_000.0},
        }
        got = MOD.extract_measurement(response, "coding")
        self.assertFalse(got["valid"])
        self.assertIn("empty_content", got["invalid_reasons"])
        self.assertIn("implausible_tps", got["invalid_reasons"])

    def test_rejects_prompt_echo(self):
        """Response that just echoes the prompt instruction is invalid."""
        text = ("Only output the code. " * 20).strip()
        response = {
            "choices": [{"text": text, "finish_reason": "length"}],
            "usage": {"completion_tokens": 100},
            "timings": {"predicted_per_second": 20.0, "draft_n": 80, "draft_n_accepted": 70},
        }
        got = MOD.extract_measurement(response, "coding")
        self.assertFalse(got["valid"])
        self.assertIn("prompt_echo", got["invalid_reasons"])

    def test_accepts_compilable_fibonacci(self):
        """Compilable Python with fibonacci definition is valid."""
        text = "```python\ndef fibonacci(n, memo=None):\n    return n\n```"
        response = {
            "choices": [{"text": text, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 20},
            "timings": {"predicted_per_second": 20.0},
        }
        got = MOD.extract_measurement(response, "coding")
        self.assertTrue(got["valid"])

    def test_rejects_coding_length_finish(self):
        """Coding prompt that hits length limit (not stop) is invalid."""
        text = "```python\ndef fibonacci(n):\n    return n\n```"
        response = {
            "choices": [{"text": text, "finish_reason": "length"}],
            "usage": {"completion_tokens": 20},
            "timings": {"predicted_per_second": 20.0},
        }
        got = MOD.extract_measurement(response, "coding")
        self.assertFalse(got["valid"])
        self.assertIn("coding_not_stopped", got["invalid_reasons"])

    def test_rejects_math_wrong_answer(self):
        """Math response with wrong answer is invalid."""
        text = "The answer is 3.7 hours."
        response = {
            "choices": [{"text": text, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 15},
            "timings": {"predicted_per_second": 20.0},
        }
        got = MOD.extract_measurement(response, "math")
        self.assertFalse(got["valid"])
        self.assertIn("math_wrong_answer", got["invalid_reasons"])

    def test_accepts_math_correct_answer(self):
        """Math response containing 2.4 hours is valid."""
        text = "The solution takes 2.4 hours to complete."
        response = {
            "choices": [{"text": text, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 15},
            "timings": {"predicted_per_second": 20.0},
        }
        got = MOD.extract_measurement(response, "math")
        self.assertTrue(got["valid"])

    def test_rejects_spec_no_drafts(self):
        """Speculative mode (MTP/DFlash) with draft_n=0 is invalid."""
        text = "Some valid code output here."
        response = {
            "choices": [{"text": text, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 20},
            "timings": {"predicted_per_second": 20.0, "draft_n": 0, "draft_n_accepted": 0},
        }
        got = MOD.extract_measurement(response, "coding", "mtp")
        self.assertFalse(got["valid"])
        self.assertIn("spec_no_drafts", got["invalid_reasons"])
        self.assertEqual(got["draft_n"], 0)


class TestProvenance(unittest.TestCase):
    """Tests for executable provenance verification."""

    def test_provenance_record_structure(self):
        """Provenance record should contain all required fields."""
        record = MOD.make_provenance_record(
            server_path="/fake/path/to/server",
            model_path="/fake/model.gguf",
            draft_path=None,
            device_line="test device",
            command=["test"],
            environment={"TEST": "1"},
        )
        self.assertIn("server_version", record)
        self.assertIn("server_sha256", record)
        self.assertIn("source_head", record)
        self.assertIn("model_sha256", record)
        self.assertIn("device", record)
        self.assertIn("command", record)


class TestRepeatedRequestsOutputArg(unittest.TestCase):
    """repeated_requests.py must accept --output argument."""

    def test_repeated_requests_accepts_output_arg(self):
        """--output must be a valid argument."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(HERE / "repeated_requests.py"), "--help"],
            capture_output=True, text=True,
        )
        self.assertIn("--output", result.stdout)


class TestConcurrentRequestSpecs(unittest.TestCase):
    """concurrent_request_specs must return one coding and one math request."""

    def test_concurrent_always_submits_both_prompt_types(self):
        """Even with prompt=coding, concurrent must return coding+math."""
        # Import repeated_requests to access the helper
        import importlib.util as iu
        rr_spec = iu.spec_from_file_location("repeated_requests", HERE / "repeated_requests.py")
        rr_mod = iu.module_from_spec(rr_spec)
        rr_spec.loader.exec_module(rr_mod)

        specs = rr_mod.concurrent_request_specs(
            {"coding": "code prompt", "math": "math prompt"},
            512,
        )
        self.assertEqual(len(specs), 2)
        prompt_types = [s[0] for s in specs]
        self.assertIn("coding", prompt_types)
        self.assertIn("math", prompt_types)
        # Both use the same gen_tokens
        for spec in specs:
            self.assertEqual(spec[2], 512)


if __name__ == "__main__":
    unittest.main()
