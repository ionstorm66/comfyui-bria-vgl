"""Subprocess-only fake provider for cancellation tests; never contacts a service."""
import json
import os
from pathlib import Path
import runpy
import sys
import time
import types


class Payload:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Scene:
    short_description = "A fox."

    def __init__(self, payload, options):
        self.payload, self.options = payload, options

    def model_dump_json(self, **kwargs):
        return json.dumps({
            "prompt": getattr(self.payload, "prompt", None),
            "llm": self.options["llm"],
            "image_size": list(self.payload.image.size) if hasattr(self.payload, "image") else None,
            "env": os.environ.get("VGL_TEST_ENV"),
        })


def create(payload, **kwargs):
    if getattr(payload, "prompt", "") == "__hang__":
        marker = os.environ.get("VGL_TEST_STARTED")
        if marker:
            Path(marker).write_text("started")
        time.sleep(60)
    if getattr(payload, "prompt", "") == "__error__":
        raise RuntimeError("test provider failure")
    return Scene(payload, kwargs)


api = types.ModuleType("bria_vgl")
api.create = create
dialect = types.ModuleType("bria_vgl.dialects.thales")
dialect.Generate = dialect.Caption = dialect.Inspire = Payload
sys.modules.update({
    "bria_vgl": api,
    "bria_vgl.dialects": types.ModuleType("bria_vgl.dialects"),
    "bria_vgl.dialects.thales": dialect,
})
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
