"""ComfyUI nodes for Bria's Visual Generation Language."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def check_interrupted():
    # Delay GPU management imports until execution.
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


def wait_for_worker(process, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            # Check before accepting a completed result, too.
            check_interrupted()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"VGL exceeded {timeout_seconds:g} seconds. The local request was stopped. "
                    "Increase timeout_seconds or check your provider."
                )
            if process.poll() is not None:
                return process.returncode
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    finally:
        # ComfyUI's interrupt inherits BaseException: always clean up, and let
        # the original interruption propagate to the executor.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run_vgl(text, llm, seed, output_format, image, timeout_seconds, mask=None, extra_images=None):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive finite number.")
    check_interrupted()
    with tempfile.TemporaryDirectory(prefix="comfyui-vgl-") as folder:
        folder = Path(folder)
        image_path = folder / "image.png"
        if image is not None:
            image.save(image_path)
        extra_image_paths = []
        for index, reference in enumerate(extra_images or [], start=2):
            path = folder / f"image_{index}.png"
            reference.save(path)
            extra_image_paths.append(str(path))
        mask_path = folder / "mask.png"
        if mask is not None:
            mask.save(mask_path)
        request = {
            "text": text, "llm": llm, "seed": seed, "output_format": output_format,
            "image_path": str(image_path) if image is not None else None,
            "mask_path": str(mask_path) if mask is not None else None,
            "extra_image_paths": extra_image_paths,
        }
        request_path = folder / "request.json"
        response_path = folder / "response.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        check_interrupted()
        # Same Python and inherited environment as ComfyUI. Files avoid pipe
        # deadlocks when a provider produces large amounts of logging.
        with (folder / "stderr.log").open("w+b") as errors:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("vgl_worker.py")),
                 str(request_path), str(response_path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            returncode = wait_for_worker(process, timeout_seconds)
            if not response_path.exists():
                raise RuntimeError(
                    f"VGL worker exited without a result (exit code {returncode}). "
                    "Check that ComfyUI's Python can run vgl_worker.py and has the node dependencies installed."
                )
            response = json.loads(response_path.read_text(encoding="utf-8"))
        check_interrupted()
        if "error" in response:
            raise RuntimeError(response["error"])
        if returncode != 0 or not isinstance(response.get("text"), str):
            raise RuntimeError(f"VGL worker returned an invalid result (exit code {returncode}).")
        return response["text"]

PROVIDERS = {
    "Google AI Studio": ("gemini/", "gemini-3.5-flash-lite"),
    "Google Vertex AI": ("vertex_ai/", "gemini-3.5-flash-lite"),
    "AWS Bedrock": ("bedrock/", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
    "Custom (LiteLLM)": ("", ""),
}


def resolve_model(provider, model):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown VGL provider: {provider}")
    prefix, default = PROVIDERS[provider]
    model = model.strip() or default
    if not model:
        raise ValueError("Enter a full LiteLLM model ID for the custom provider.")
    if prefix and not model.startswith(prefix):
        model = prefix + model
    return model


def image_to_pil(image):
    import numpy as np
    from PIL import Image

    if len(image.shape) != 4 or image.shape[0] != 1:
        raise ValueError("VGL expects one IMAGE [1, height, width, channels]. Select one image from a batch first.")
    if image.shape[-1] not in (1, 3, 4) or min(image.shape[1:3]) < 1:
        raise ValueError("VGL requires a nonempty image with 1, 3, or 4 channels.")
    pixels = image[0].detach().cpu().float().numpy()
    if not np.isfinite(pixels).all():
        raise ValueError("VGL image contains NaN or infinite pixel values.")
    pixels = (np.clip(pixels, 0, 1) * 255).round().astype(np.uint8)
    if pixels.shape[-1] == 1:
        pixels = pixels[..., 0]
    result = Image.fromarray(pixels)
    if result.mode == "RGBA":
        background = Image.new("RGBA", result.size, "white")
        result = Image.alpha_composite(background, result)
    return result.convert("RGB")


def mask_to_pil(mask, image_size):
    """ComfyUI MASK [1,H,W] (or unbatched [H,W]) to VGL grayscale."""
    import numpy as np
    from PIL import Image

    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if len(mask.shape) != 2 or min(mask.shape) < 1:
        raise ValueError("VGL expects one MASK [1, height, width] or [height, width].")
    if (mask.shape[1], mask.shape[0]) != image_size:
        raise ValueError("Mask dimensions must match the image. Resize the mask upstream.")
    pixels = mask.detach().cpu().float().numpy()
    if not np.isfinite(pixels).all():
        raise ValueError("VGL mask contains NaN or infinite values.")
    # No inversion: ComfyUI 1 is the region to edit, just like VGL white.
    pixels = (np.clip(pixels, 0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(pixels)


def masked_image_output(image, mask):
    """FIBO2 grey_out preprocessing, before the encoder resizes/crops."""
    if mask is None:
        return None
    import numpy as np
    import torch
    from PIL import Image

    gray = Image.new("RGB", image.size, (128, 128, 128))
    binary_mask = mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
    composite = Image.composite(gray, image.convert("RGB"), binary_mask)
    pixels = np.asarray(composite, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels[None, ...].copy())


class BriaVGLPrompt:
    def __init__(self):
        # One successful result per instance; no image or credentials stored.
        self._cached_prompt = None

    def clear_cache(self):
        self._cached_prompt = None

    @staticmethod
    def cache_key(text, provider, llm, output_format, seed, image, mask=None):
        image_key = None
        if image is not None:
            image_key = (image.mode, image.size, hashlib.sha256(image.tobytes()).digest())
        routing = tuple(os.environ.get(name) for name in (
            "OPENAI_BASE_URL", "OPENAI_API_BASE", "VERTEXAI_PROJECT",
            "VERTEXAI_LOCATION", "AWS_REGION_NAME", "AWS_PROFILE",
        ))
        mask_key = None if mask is None else (mask.size, hashlib.sha256(mask.tobytes()).digest())
        return (text, provider, llm, output_format, seed, image_key, mask_key, routing)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "Type a prompt or convert this widget to an input to connect a text node. Leave blank to describe an image."}),
                "provider": (list(PROVIDERS),),
                "model": ("STRING", {"default": "", "tooltip": "Blank uses the provider default. Custom requires a full LiteLLM model ID."}),
                "output_format": (["VGL JSON", "Short description"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK", {"tooltip": "White/1 edits, black/0 keeps. Requires an image and edit text; dimensions must match."}),
                "timeout_seconds": ("INT", {"default": 180, "min": 1, "max": 3600,
                    "tooltip": "Maximum total wait, including startup and provider retries. Stop cancels the local worker."}),
                "image_2": ("IMAGE", {"tooltip": "Reference <image_2>. Connect image first. Requires edit text; no mask."}),
                "image_3": ("IMAGE", {"tooltip": "Reference <image_3>. Connect image and image_2 first."}),
                "image_4": ("IMAGE", {"tooltip": "Reference <image_4>. Connect all earlier image inputs first."}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "masked_image")
    OUTPUT_TOOLTIPS = (
        "Generated VGL JSON or short description.",
        "Source image with mask values >=128 filled with RGB(128,128,128). Without a mask, passes through the original image. Connect to FIBO2 image_1.",
    )
    FUNCTION = "generate"
    CATEGORY = "Bria/VGL"
    DESCRIPTION = "Create a VGL prompt from text, an image, or both using the selected LLM provider. Credentials come from the ComfyUI process environment."

    def generate(self, text, provider, model="", output_format="VGL JSON", seed=0, image=None, timeout_seconds=180, mask=None,
                 image_2=None, image_3=None, image_4=None):
        check_interrupted()
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number.")
        text = text.strip()
        references = [image, image_2, image_3, image_4]
        for index, reference in enumerate(references[1:], start=1):
            if reference is not None and references[index - 1] is None:
                raise ValueError("Connect reference images in order: image, image_2, image_3, image_4; do not skip inputs.")
        extra_images = [image_to_pil(ref) for ref in references[1:] if ref is not None]
        if extra_images and (not text or mask is not None):
            raise ValueError("Multiple reference images require nonempty edit text and cannot be combined with a mask.")
        if mask is not None and (image is None or not text):
            raise ValueError("A mask requires an image and nonempty edit text.")
        if not text and image is None:
            raise ValueError("VGL needs text, an image, or both.")
        if output_format not in ("VGL JSON", "Short description"):
            raise ValueError(f"Unknown output format: {output_format}")
        llm = resolve_model(provider, model)
        pil_image = image_to_pil(image) if image is not None else None
        pil_mask = mask_to_pil(mask, pil_image.size) if mask is not None else None
        masked_image = masked_image_output(pil_image, pil_mask) if pil_mask is not None else image
        key = self.cache_key(text, provider, llm, output_format, seed, pil_image, pil_mask)
        key += (tuple((ref.size, hashlib.sha256(ref.tobytes()).digest()) for ref in extra_images),)
        check_interrupted()
        cached = self._cached_prompt
        if cached is not None and cached[0] == key:
            return (cached[1], masked_image)
        options = {"extra_images": extra_images} if extra_images else {}
        result = run_vgl(text, llm, seed, output_format, pil_image, timeout_seconds, mask=pil_mask, **options)
        check_interrupted()
        self._cached_prompt = (key, result)
        return (result, masked_image)


class BriaVGLTextInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": "", "multiline": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "value"
    CATEGORY = "Bria/VGL"

    def value(self, text):
        return (text,)


def local_vgl_dependencies():
    try:
        from bria_vgl.dialects.thales import Generate, Caption, Inspire, EditImageWithMask
        from bria_vgl.dialects.thales.model import ThalesVGL
        from bria_vgl.prompting import render_prompts
    except ImportError as exc:
        raise RuntimeError("Install this node's requirements.txt using ComfyUI's Python, then restart ComfyUI.") from exc
    return Generate, Caption, Inspire, EditImageWithMask, ThalesVGL, render_prompts


class BriaVGLPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": "", "multiline": True})},
                "optional": {"image": ("IMAGE",), "mask": ("MASK",)}}

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("system_prompt", "prompt", "image", "masked_image", "require_edit_instruction")
    FUNCTION = "prepare"
    CATEGORY = "Bria/VGL"
    DESCRIPTION = "Prepare Bria's VGL instructions for ComfyUI Generate Text. No provider or API key required."

    def prepare(self, text, image=None, mask=None):
        check_interrupted()
        text = text.strip()
        if mask is not None and (image is None or not text):
            raise ValueError("A mask requires an image and nonempty edit text.")
        if not text and image is None:
            raise ValueError("VGL needs text, an image, or both.")
        Generate, Caption, Inspire, EditImageWithMask, model, render = local_vgl_dependencies()
        pil_image = image_to_pil(image) if image is not None else None
        pil_mask = mask_to_pil(mask, pil_image.size) if mask is not None else None
        if pil_mask is not None:
            payload = EditImageWithMask(prompt=text, image=pil_image, mask=pil_mask)
        elif pil_image is None:
            payload = Generate(prompt=text)
        elif text:
            payload = Inspire(prompt=text, image=pil_image)
        else:
            payload = Caption(image=pil_image)
        request = payload.to_request()
        system, prompt = render(request.prompts, request.context,
                                system_template=request.system_prompt, final_template=request.final_prompt)
        schema = model.model_json_schema()
        if mask is not None:
            schema.setdefault("required", []).append("edit_instruction")
        system += ("\n\nReturn exactly one JSON object matching the following JSON Schema. "
                   "Do not include Markdown fences, commentary, or reasoning.\n" + json.dumps(schema))
        if mask is not None:
            system += "\nInclude a nonempty edit_instruction describing the requested change."
        prepared_image = None
        if request.images:
            import numpy as np
            import torch
            prepared_image = torch.from_numpy(
                np.asarray(request.images[0].convert("RGB"), dtype=np.float32)[None, ...].copy() / 255.0)
        reference = masked_image_output(pil_image, pil_mask) if pil_mask is not None else image
        check_interrupted()
        return system, prompt, prepared_image, reference, mask is not None


class BriaVGLValidate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"generated_text": ("STRING", {"forceInput": True})},
                "optional": {"require_edit_instruction": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("vgl_json", "short_description")
    FUNCTION = "validate"
    CATEGORY = "Bria/VGL"
    DESCRIPTION = "Validate local model output against Bria's Thales VGL model before using it as a prompt."

    def validate(self, generated_text, require_edit_instruction=False):
        check_interrupted()
        model = local_vgl_dependencies()[4]
        text = generated_text.strip()
        # Accept a single enclosing JSON fence, but never guess where prose ends.
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].lower() in ("```json", "```") and lines[-1] == "```":
            text = "\n".join(lines[1:-1])
        try:
            scene = model.model_validate_json(text, strict=True)
            if not scene.short_description.strip():
                raise ValueError("short_description must not be empty")
            if require_edit_instruction and not (scene.edit_instruction or "").strip():
                raise ValueError("masked edits require a nonempty edit_instruction")
        except ValueError as exc:
            raise ValueError("Invalid VGL from Generate Text. Check the model, prompt, and max_length "
                             f"(the response may be truncated). Details: {exc}") from exc
        check_interrupted()
        return scene.model_dump_json(indent=2, exclude_none=True), scene.short_description


NODE_CLASS_MAPPINGS = {
    "BriaVGLPrepare": BriaVGLPrepare,
    "BriaVGLValidate": BriaVGLValidate,
    "BriaVGLPrompt": BriaVGLPrompt,
    "BriaVGLTextInput": BriaVGLTextInput,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BriaVGLPrepare": "Bria VGL Prepare (Local)",
    "BriaVGLValidate": "Bria VGL Validate",
    "BriaVGLPrompt": "Bria VGL Prompt",
    "BriaVGLTextInput": "Bria VGL Text Input",
}
