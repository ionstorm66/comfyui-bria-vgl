import json
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import BriaVGLPrompt, BriaVGLTextInput, image_to_pil, resolve_model
import vgl_worker


class Tensor:
    """Small tensor protocol double; no GPU or PyTorch needed."""
    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape = self.data.shape

    def __getitem__(self, key):
        return Tensor(self.data[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.data


class Payload:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Generate(Payload):
    pass


class Caption(Payload):
    pass


class Inspire(Payload):
    pass


class Scene:
    short_description = "A red fox."

    def model_dump_json(self, **kwargs):
        return json.dumps({"short_description": self.short_description})


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        api = types.ModuleType("bria_vgl")
        def create(payload, **kwargs):
            self.calls.append((payload, kwargs))
            return Scene()
        api.create = create
        dialect = types.ModuleType("bria_vgl.dialects.thales")
        dialect.Generate, dialect.Caption, dialect.Inspire = Generate, Caption, Inspire
        self.modules = patch.dict(sys.modules, {
            "bria_vgl": api,
            "bria_vgl.dialects": types.ModuleType("bria_vgl.dialects"),
            "bria_vgl.dialects.thales": dialect,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        # Real worker logic with a fake provider; process tests are separate.
        runner = patch("nodes.run_vgl", side_effect=lambda text, llm, seed, output_format, image, timeout_seconds:
                       vgl_worker.generate(text, llm, seed, output_format, image))
        runner.start()
        self.addCleanup(runner.stop)

    def test_text_json_and_provider_forwarding(self):
        result = BriaVGLPrompt().generate(" fox ", "Google Vertex AI", seed=42)
        self.assertEqual(json.loads(result[0])["short_description"], "A red fox.")
        payload, options = self.calls[0]
        self.assertIsInstance(payload, Generate)
        self.assertEqual(payload.prompt, "fox")
        self.assertEqual(options, {"llm": "vertex_ai/gemini-3.5-flash-lite", "seed": 42})

    def test_image_caption_and_inspiration(self):
        image = Tensor(np.zeros((1, 3, 4, 3)))
        for text, expected in [("", Caption), ("fox", Inspire)]:
            with self.subTest(text=text):
                result = BriaVGLPrompt().generate(text, "Google AI Studio", image=image, output_format="Short description")
                self.assertEqual(result, ("A red fox.",))
                payload = self.calls[-1][0]
                self.assertIsInstance(payload, expected)
                self.assertEqual(payload.image.size, (4, 3))
                self.assertEqual(payload.image.mode, "RGB")

    def test_empty_rejected_without_provider_call(self):
        with self.assertRaisesRegex(ValueError, "needs text"):
            BriaVGLPrompt().generate("  ", "Google AI Studio")
        self.assertFalse(self.calls)

    def test_model_selection(self):
        self.assertEqual(resolve_model("AWS Bedrock", ""), "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        self.assertEqual(resolve_model("Google AI Studio", "gemini/test"), "gemini/test")
        self.assertEqual(resolve_model("Google AI Studio", "test"), "gemini/test")
        self.assertEqual(resolve_model("Custom (LiteLLM)", "openai/test"), "openai/test")
        with self.assertRaises(ValueError):
            resolve_model("Custom (LiteLLM)", "")

    def test_conversion_clipping_and_alpha(self):
        image = Tensor([[[[-0.1, 0.5, 1.2]]]])
        self.assertEqual(image_to_pil(image).getpixel((0, 0)), (0, 128, 255))
        image = Tensor(np.zeros((1, 1, 1, 4)))
        self.assertEqual(image_to_pil(image).getpixel((0, 0)), (255, 255, 255))

    def test_invalid_images(self):
        for array in [np.zeros((2, 1, 1, 3)), np.zeros((1, 1, 3)), np.zeros((1, 1, 1, 2)), np.full((1, 1, 1, 3), np.nan)]:
            with self.subTest(shape=array.shape):
                with self.assertRaises(ValueError):
                    image_to_pil(Tensor(array))

    def test_provider_errors_propagate(self):
        with patch("bria_vgl.create", side_effect=RuntimeError("quota exhausted")):
            with self.assertRaisesRegex(RuntimeError, "quota exhausted"):
                BriaVGLPrompt().generate("fox", "Google AI Studio")

    def test_text_input(self):
        self.assertEqual(BriaVGLTextInput().value("hello\nworld"), ("hello\nworld",))


if __name__ == "__main__":
    unittest.main()
