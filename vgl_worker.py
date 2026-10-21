"""Isolated VGL request process. Does not import ComfyUI or initialize its GPU."""

import json
from pathlib import Path
import sys


def generate(text, llm, seed, output_format, image=None, mask=None, extra_images=None):
    try:
        import bria_vgl
        from bria_vgl.dialects.thales import Caption, Generate, Inspire, EditImageWithMask
    except ImportError as exc:
        raise RuntimeError(
            "Install this node's requirements.txt using ComfyUI's Python, then restart ComfyUI."
        ) from exc
    if extra_images:
        from bria_vgl.dialects.thales import EditImages
        if image is None or not text.strip() or mask is not None or len(extra_images) > 3:
            raise ValueError("Multi-image edits require 2–4 images, nonempty text, and no mask.")
        payload = EditImages(prompt=text, images=[image, *extra_images])
    elif mask is not None:
        if image is None or not text.strip():
            raise ValueError("A mask requires an image and nonempty edit text.")
        payload = EditImageWithMask(prompt=text, image=image, mask=mask)
    elif image is None:
        payload = Generate(prompt=text)
    elif text:
        payload = Inspire(prompt=text, image=image)
    else:
        payload = Caption(image=image)
    operation = bria_vgl.modify if mask is not None or extra_images else bria_vgl.create
    scene = operation(payload, llm=llm, seed=seed)
    if output_format == "VGL JSON":
        return scene.model_dump_json(indent=2, exclude_none=True)
    result = scene.short_description
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError("VGL returned no short description. Try the VGL JSON output format.")
    return result


def main():
    response_path = Path(sys.argv[2])
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        image_path = request.pop("image_path")
        mask_path = request.pop("mask_path", None)
        extra_image_paths = request.pop("extra_image_paths", [])
        extra_images = []
        for path in extra_image_paths:
            from PIL import Image
            with Image.open(path) as source:
                extra_images.append(source.convert("RGB"))
        image = None
        if image_path is not None:
            from PIL import Image
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        mask = None
        if mask_path is not None:
            from PIL import Image
            with Image.open(mask_path) as source:
                mask = source.convert("L")
        result = generate(**request, image=image, mask=mask, extra_images=extra_images)
        response = {"text": result}
        exit_code = 0
    except Exception as exc:
        response = {"error": f"{type(exc).__name__}: {exc}"}
        exit_code = 1
    response_path.write_text(json.dumps(response), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
