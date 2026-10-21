"""ComfyUI nodes for Bria's Visual Generation Language."""

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


def run_vgl(text, llm, seed, output_format, image, timeout_seconds):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive finite number.")
    check_interrupted()
    with tempfile.TemporaryDirectory(prefix="comfyui-vgl-") as folder:
        folder = Path(folder)
        image_path = folder / "image.png"
        if image is not None:
            image.save(image_path)
        request = {
            "text": text, "llm": llm, "seed": seed, "output_format": output_format,
            "image_path": str(image_path) if image is not None else None,
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


class BriaVGLPrompt:
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
                "timeout_seconds": ("INT", {"default": 180, "min": 1, "max": 3600,
                    "tooltip": "Maximum total wait, including startup and provider retries. Stop cancels the local worker."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "Bria/VGL"
    DESCRIPTION = "Create a VGL prompt from text, an image, or both using the selected LLM provider. Credentials come from the ComfyUI process environment."

    def generate(self, text, provider, model="", output_format="VGL JSON", seed=0, image=None, timeout_seconds=180):
        text = text.strip()
        if not text and image is None:
            raise ValueError("VGL needs text, an image, or both.")
        if output_format not in ("VGL JSON", "Short description"):
            raise ValueError(f"Unknown output format: {output_format}")
        llm = resolve_model(provider, model)
        pil_image = image_to_pil(image) if image is not None else None
        result = run_vgl(text, llm, seed, output_format, pil_image, timeout_seconds)
        return (result,)


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


NODE_CLASS_MAPPINGS = {
    "BriaVGLPrompt": BriaVGLPrompt,
    "BriaVGLTextInput": BriaVGLTextInput,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BriaVGLPrompt": "Bria VGL Prompt",
    "BriaVGLTextInput": "Bria VGL Text Input",
}
