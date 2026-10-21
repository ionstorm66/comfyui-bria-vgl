"""Integration checks with installed bria-vgl; no model or provider calls."""
import importlib.util
import json
import sys
import types
import unittest
from unittest.mock import patch

from test_nodes import Tensor
from nodes import BriaVGLPrepare, BriaVGLValidate


@unittest.skipUnless(importlib.util.find_spec("bria_vgl"), "Install bria-vgl for local integration tests")
class LocalTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch("nodes.check_interrupted")
        self.guard.start()
        self.addCleanup(self.guard.stop)
        torch = types.ModuleType("torch")
        torch.from_numpy = lambda array: array
        self.modules = patch.dict(sys.modules, {"torch": torch})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.scene = {
            "short_description": "A red ball.", "background_setting": "A room.",
            "lighting": {"conditions": "daylight", "direction": "left"},
            "aesthetics": {"composition": "centered", "color_scheme": "warm", "mood_atmosphere": "calm"},
            "context": "Still life",
        }

    def test_text_and_images_use_real_templates(self):
        import bria_vgl
        with patch.object(bria_vgl, "create", side_effect=AssertionError("No API calls")):
            system, prompt, image, reference, edit = BriaVGLPrepare().prepare("a red ball")
            self.assertIn("a red ball", prompt)
            self.assertIn('"$defs"', system)
            self.assertNotIn("{llm_input_schema}", system)
            self.assertIsNone(image)
            self.assertIsNone(reference)
            self.assertFalse(edit)
            source = Tensor([[[[1., 0., 0.], [0., 1., 0.]]]])
            for text in ("", "a red ball"):
                result = BriaVGLPrepare().prepare(text, source)
                self.assertEqual(result[2].shape, (1, 1, 2, 3))
                self.assertIs(result[3], source)
            result = BriaVGLPrepare().prepare("replace ball", source, Tensor([[[0.25, 1.]]]))
            self.assertTrue(result[4])
            # Soft mask for vision; binary gray-out reference for FIBO.
            self.assertGreater(result[2][0, 0, 0, 1], 0)
            self.assertEqual(result[3][0, 0, 0, 1], 0)
            self.assertAlmostEqual(float(result[3][0, 0, 1, 1]), 128 / 255)

    def test_bad_inputs(self):
        for args in ({"text": " "}, {"text": "edit", "mask": Tensor([[[1.]]])}):
            with self.assertRaises(ValueError):
                BriaVGLPrepare().prepare(**args)

    def test_worker_uses_real_multi_image_payload(self):
        from PIL import Image
        from bria_vgl.dialects.thales import EditImages
        import vgl_worker
        images = [Image.new("RGB", (n + 2, 3)) for n in range(4)]
        with patch("bria_vgl.modify") as modify, patch("bria_vgl.create", side_effect=AssertionError("Must modify")):
            modify.return_value.model_dump_json.return_value = '{}'
            vgl_worker.generate("combine <image_1> and <image_4>", "test", 0, "VGL JSON",
                                images[0], extra_images=images[1:])
            payload = modify.call_args.args[0]
            self.assertIsInstance(payload, EditImages)
            self.assertEqual(payload.images, images)
            self.assertEqual(len(payload.to_request().images), 4)

    def test_validation_and_fences(self):
        raw = json.dumps(self.scene)
        for value in (raw, "```json\n" + raw + "\n```"):
            output, short = BriaVGLValidate().validate(value)
            self.assertEqual(json.loads(output), self.scene)
            self.assertEqual(short, self.scene["short_description"])

    def test_invalid_and_incomplete_results(self):
        for value in ('{}', '[]', '{', 'Here is JSON: ' + json.dumps(self.scene),
                      json.dumps(self.scene | {"lighting": "daylight"}),
                      json.dumps(self.scene | {"short_description": " "})):
            with self.assertRaisesRegex(ValueError, "Invalid VGL"):
                BriaVGLValidate().validate(value)
        with self.assertRaisesRegex(ValueError, "edit_instruction"):
            BriaVGLValidate().validate(json.dumps(self.scene), True)
        self.scene["edit_instruction"] = "Replace the ball."
        self.assertIn("edit_instruction", BriaVGLValidate().validate(json.dumps(self.scene), True)[0])
