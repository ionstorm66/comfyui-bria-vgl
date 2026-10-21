import json
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import BriaVGLPrompt, BriaVGLTextInput, image_to_pil, mask_to_pil, resolve_model
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


class EditImageWithMask(Payload):
    pass


class Scene:
    short_description = "A red fox."

    def model_dump_json(self, **kwargs):
        return json.dumps({"short_description": self.short_description})


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        interrupt = patch("nodes.check_interrupted")
        self.interrupt = interrupt.start()
        self.addCleanup(interrupt.stop)
        api = types.ModuleType("bria_vgl")
        # Test IMAGE layout without installing ComfyUI/PyTorch.
        torch = types.ModuleType("torch")
        torch.from_numpy = lambda array: array
        def create(payload, **kwargs):
            self.calls.append((payload, kwargs))
            return Scene()
        api.create = create
        api.modify = create
        dialect = types.ModuleType("bria_vgl.dialects.thales")
        dialect.Generate, dialect.Caption, dialect.Inspire = Generate, Caption, Inspire
        dialect.EditImageWithMask = EditImageWithMask
        self.modules = patch.dict(sys.modules, {
            "torch": torch,
            "bria_vgl": api,
            "bria_vgl.dialects": types.ModuleType("bria_vgl.dialects"),
            "bria_vgl.dialects.thales": dialect,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        # Real worker logic with a fake provider; process tests are separate.
        runner = patch("nodes.run_vgl", side_effect=lambda text, llm, seed, output_format, image, timeout_seconds, mask=None:
                       vgl_worker.generate(text, llm, seed, output_format, image, mask))
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
                self.assertEqual(result, ("A red fox.", None))
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

    def test_mask_conversion(self):
        mask = Tensor([[[-0.2, 0, 0.5, 1, 1.2]]])
        result = mask_to_pil(mask, (5, 1))
        self.assertEqual(result.mode, "L")
        self.assertEqual(list(result.getdata()), [0, 0, 128, 255, 255])
        self.assertEqual(result.tobytes(), mask_to_pil(mask[0], (5, 1)).tobytes())

    def test_masked_image_matches_fibo_threshold_and_cache(self):
        node = BriaVGLPrompt()
        image = Tensor(np.array([[[[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                   [1, 1, 0], [0, 1, 1]]]], dtype=np.float32))
        original = image.data.copy()
        mask = Tensor([[[0, 127 / 255, 128 / 255, 0.5, 1]]])
        result = node.generate("edit", "Google AI Studio", image=image, mask=mask)
        output = result[1]
        self.assertEqual(output.shape, (1, 1, 5, 3))
        self.assertEqual(output.dtype, np.float32)
        expected = np.array([[[255, 0, 0], [0, 255, 0], [128, 128, 128],
                              [128, 128, 128], [128, 128, 128]]], dtype=np.float32) / 255
        np.testing.assert_array_equal(output[0], expected)
        np.testing.assert_array_equal(image.data, original)
        # Prompting still receives soft grayscale, not a binary mask.
        self.assertEqual(list(self.calls[-1][0].mask.tobytes()), [0, 127, 128, 128, 255])
        output[:] = 0
        cached = node.generate("edit", "Google AI Studio", image=image, mask=mask)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(cached[0], result[0])
        np.testing.assert_array_equal(cached[1][0], expected)
        mask.data[0, 0, 0] = 1
        changed = node.generate("edit", "Google AI Studio", image=image, mask=mask)
        self.assertEqual(len(self.calls), 2)
        self.assertAlmostEqual(float(changed[1][0, 0, 0, 0]), 128 / 255)

    def test_no_mask_has_empty_image_output(self):
        node = BriaVGLPrompt()
        self.assertIsNone(node.generate("fox", "Google AI Studio")[1])
        self.assertIsNone(node.generate("fox", "Google AI Studio")[1])
        self.assertEqual(BriaVGLPrompt.RETURN_TYPES, ("STRING", "IMAGE"))
        self.assertEqual(BriaVGLTextInput.RETURN_TYPES, ("STRING",))

    def test_invalid_masks(self):
        for array in (np.zeros((2, 2, 3)), np.zeros((1, 2, 3, 1)),
                      np.zeros((1, 0, 3)), np.full((1, 2, 3), np.nan),
                      np.full((1, 2, 3), np.inf), np.zeros((1, 3, 2))):
            with self.subTest(shape=array.shape):
                with self.assertRaises(ValueError):
                    mask_to_pil(Tensor(array), (3, 2))

    def test_mask_requires_text_and_image(self):
        mask = Tensor(np.ones((1, 2, 3)))
        for text, image in (("edit", None), (" ", Tensor(np.zeros((1, 2, 3, 3))))):
            with self.assertRaisesRegex(ValueError, "mask requires"):
                BriaVGLPrompt().generate(text, "Google AI Studio", image=image, mask=mask)
        self.assertFalse(self.calls)

    def test_mask_uses_modify_and_invalidates_cache(self):
        node = BriaVGLPrompt()
        image = Tensor(np.zeros((1, 2, 3, 3)))
        mask = Tensor(np.ones((1, 2, 3)))
        with patch("bria_vgl.create", side_effect=AssertionError("Must use modify")):
            node.generate("edit", "Google AI Studio", image=image, mask=mask)
            payload = self.calls[-1][0]
            self.assertIsInstance(payload, EditImageWithMask)
            self.assertEqual(payload.mask.mode, "L")
            self.assertEqual(payload.mask.size, payload.image.size)
            node.generate("edit", "Google AI Studio", image=image, mask=Tensor(mask.data.copy()))
            self.assertEqual(len(self.calls), 1)
            mask.data[0, 0, 0] = 0
            node.generate("edit", "Google AI Studio", image=image, mask=mask)
            self.assertEqual(len(self.calls), 2)
        node.generate("edit", "Google AI Studio", image=image)
        self.assertEqual(len(self.calls), 3)
        self.assertIsInstance(self.calls[-1][0], Inspire)

    def test_text_input(self):
        self.assertEqual(BriaVGLTextInput().value("hello\nworld"), ("hello\nworld",))

    def test_same_inputs_and_seed_reuse_result(self):
        node = BriaVGLPrompt()
        first = node.generate("fox", "Google AI Studio", seed=7)
        self.assertEqual(first, node.generate("fox", "Google AI Studio", seed=7))
        self.assertEqual(len(self.calls), 1)
        # Timeout is a wait limit, not a generation input.
        self.assertEqual(first, node.generate("fox", "Google AI Studio", seed=7, timeout_seconds=90))
        self.assertEqual(len(self.calls), 1)

    def test_each_generation_input_invalidates_cache(self):
        for change in (
            {"text": "owl"}, {"seed": 8}, {"provider": "Google Vertex AI"},
            {"model": "another-model"}, {"output_format": "Short description"},
        ):
            with self.subTest(change=change):
                node = BriaVGLPrompt()
                args = {"text": "fox", "provider": "Google AI Studio", "seed": 7}
                before = len(self.calls)
                node.generate(**args)
                node.generate(**(args | change))
                self.assertEqual(len(self.calls), before + 2)

    def test_images_use_pixels_not_tensor_identity(self):
        node = BriaVGLPrompt()
        image = Tensor(np.zeros((1, 2, 3, 3)))
        node.generate("fox", "Google AI Studio", image=image)
        node.generate("fox", "Google AI Studio", image=Tensor(image.data.copy()))
        self.assertEqual(len(self.calls), 1)
        image.data[0, 0, 0, 0] = 1
        node.generate("fox", "Google AI Studio", image=image)
        self.assertEqual(len(self.calls), 2)
        node.generate("fox", "Google AI Studio", image=Tensor(np.zeros((1, 3, 2, 3))))
        node.generate("fox", "Google AI Studio")
        self.assertEqual(len(self.calls), 4)

    def test_changed_endpoint_invalidates_cache(self):
        import os
        node = BriaVGLPrompt()
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://first/v1"}):
            node.generate("fox", "Custom (LiteLLM)", model="openai/test")
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://second/v1"}):
            node.generate("fox", "Custom (LiteLLM)", model="openai/test")
        self.assertEqual(len(self.calls), 2)

    def test_failures_and_timeouts_are_not_cached(self):
        for failure in (RuntimeError("failed"), TimeoutError("timed out")):
            with self.subTest(failure=failure):
                node = BriaVGLPrompt()
                with patch("nodes.run_vgl", side_effect=[failure, "success"]) as run:
                    with self.assertRaises(type(failure)):
                        node.generate("fox", "Google AI Studio")
                    self.assertEqual(node.generate("fox", "Google AI Studio"), ("success", None))
                    self.assertEqual(node.generate("fox", "Google AI Studio"), ("success", None))
                    self.assertEqual(run.call_count, 2)

    def test_interrupted_result_is_not_cached(self):
        class Interrupted(BaseException):
            pass
        node = BriaVGLPrompt()
        self.interrupt.side_effect = [None, None, Interrupted()]
        with self.assertRaises(Interrupted):
            node.generate("fox", "Google AI Studio")
        self.interrupt.side_effect = None
        node.generate("fox", "Google AI Studio")
        self.assertEqual(len(self.calls), 2)
        self.interrupt.side_effect = Interrupted()
        with self.assertRaises(Interrupted):
            node.generate("fox", "Google AI Studio")

    def test_clear_cache_and_instance_isolation(self):
        first, second = BriaVGLPrompt(), BriaVGLPrompt()
        first.generate("fox", "Google AI Studio")
        second.generate("fox", "Google AI Studio")
        first.clear_cache()
        first.generate("fox", "Google AI Studio")
        self.assertEqual(len(self.calls), 3)


if __name__ == "__main__":
    unittest.main()
