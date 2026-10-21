import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nodes


class InterruptProcessingException(BaseException):
    pass


class InterruptionTests(unittest.TestCase):
    def setUp(self):
        self.processes = []
        self.request_dirs = []
        real_popen = subprocess.Popen

        def launch(args, **kwargs):
            # Run the real worker with only its provider replaced.
            self.request_dirs.append(Path(args[-2]).parent)
            args = [args[0], str(Path(__file__).with_name("fake_provider.py")), *args[1:]]
            process = real_popen(args, **kwargs)
            self.processes.append(process)
            return process

        launcher = patch("nodes.subprocess.Popen", side_effect=launch)
        launcher.start()
        self.addCleanup(launcher.stop)
        # Use the production check_interrupted function against ComfyUI's
        # module contract, including its BaseException interruption type.
        self.comfy_management = types.ModuleType("comfy.model_management")
        self.comfy_management.throw_exception_if_processing_interrupted = lambda: None
        modules = patch.dict(sys.modules, {
            "comfy": types.ModuleType("comfy"),
            "comfy.model_management": self.comfy_management,
        })
        modules.start()
        self.addCleanup(modules.stop)
        self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    def run_request(self, text="fox", image=None, timeout=10):
        return nodes.run_vgl(text, "openai/test", 7, "VGL JSON", image, timeout)

    def assert_cleaned(self):
        self.assertTrue(self.processes)
        self.assertTrue(all(p.poll() is not None for p in self.processes))
        self.assertTrue(all(not path.exists() for path in self.request_dirs))

    def test_success_roundtrip_image_and_environment(self):
        with patch.dict(os.environ, {"VGL_TEST_ENV": "inherited"}):
            output = json.loads(self.run_request(image=Image.new("RGB", (3, 2))))
        self.assertEqual(output, {
            "prompt": "fox", "llm": "openai/test", "image_size": [3, 2], "env": "inherited",
        })
        self.assert_cleaned()

    def test_provider_error_roundtrip(self):
        with self.assertRaisesRegex(RuntimeError, "test provider failure"):
            self.run_request("__error__")
        self.assert_cleaned()

    def test_interrupt_stalled_provider_then_run_again(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "started"
            interrupted_at = []
            def interrupt_when_provider_starts():
                if marker.exists():
                    interrupted_at.append(time.monotonic())
                    raise InterruptProcessingException()
            self.comfy_management.throw_exception_if_processing_interrupted = interrupt_when_provider_starts
            with patch.dict(os.environ, {"VGL_TEST_STARTED": str(marker)}):
                with self.assertRaises(InterruptProcessingException):
                    self.run_request("__hang__")
            self.assertLess(time.monotonic() - interrupted_at[0], 3)
        self.assert_cleaned()
        self.comfy_management.throw_exception_if_processing_interrupted = lambda: None
        self.assertEqual(json.loads(self.run_request())["prompt"], "fox")
        self.assert_cleaned()

    def test_timeout_terminates_and_cleans_worker(self):
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "exceeded"):
            self.run_request("__hang__", timeout=0.3)
        self.assertLess(time.monotonic() - started, 4)
        self.assert_cleaned()

    def test_pending_interrupt_never_starts_worker(self):
        def interrupt():
            raise InterruptProcessingException()
        self.comfy_management.throw_exception_if_processing_interrupted = interrupt
        with self.assertRaises(InterruptProcessingException):
            self.run_request()
        self.assertFalse(self.processes)

    def test_interrupt_wins_over_completed_result(self):
        class Completed:
            returncode = 0
            def poll(self):
                return 0
        def interrupt():
            raise InterruptProcessingException()
        self.comfy_management.throw_exception_if_processing_interrupted = interrupt
        with self.assertRaises(InterruptProcessingException):
            nodes.wait_for_worker(Completed(), 10)

    def test_bad_timeout_never_starts_worker(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    self.run_request(timeout=timeout)
        self.assertFalse(self.processes)


if __name__ == "__main__":
    unittest.main()
