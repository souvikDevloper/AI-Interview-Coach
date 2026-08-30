import os
import unittest
from unittest.mock import patch

from speech_recognition.offline import _candidate_configs


class SpeechConfigurationTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_cpu(self):
        with patch.dict(os.environ, {}, clear=True), patch("os.cpu_count", return_value=8):
            configs = _candidate_configs()
        self.assertEqual(configs[0]["device"], "cuda")
        self.assertEqual(configs[0]["compute_type"], "float16")
        self.assertEqual(configs[1]["device"], "cpu")
        self.assertEqual(configs[1]["compute_type"], "int8")
        self.assertEqual(configs[1]["cpu_threads"], 4)

    def test_explicit_cpu(self):
        with patch.dict(
            os.environ,
            {"WHISPER_DEVICE": "cpu", "WHISPER_COMPUTE": "float32"},
            clear=True,
        ):
            configs = _candidate_configs()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["device"], "cpu")
        self.assertEqual(configs[0]["compute_type"], "float32")

    def test_cuda_can_disable_fallback(self):
        with patch.dict(
            os.environ,
            {"WHISPER_DEVICE": "cuda", "WHISPER_ALLOW_FALLBACK": "0"},
            clear=True,
        ):
            configs = _candidate_configs()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["device"], "cuda")

    def test_invalid_device_is_rejected(self):
        with patch.dict(os.environ, {"WHISPER_DEVICE": "tpu"}, clear=True):
            with self.assertRaises(ValueError):
                _candidate_configs()


if __name__ == "__main__":
    unittest.main()
