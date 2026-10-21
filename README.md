# ComfyUI Bria VGL

Create prompts using [Bria-AI/vgl](https://github.com/Bria-AI/vgl).
Includes **Bria VGL Prompt** and **Bria VGL Text Input**, under `Bria/VGL`.
Requires Python 3.10+ and credentials for your selected provider.

## Install

Copy this folder into `ComfyUI/custom_nodes/comfyui-bria-vgl`.
Install dependencies with the Python interpreter that runs ComfyUI, then restart it:

```powershell
# Standard install: run from ComfyUI with its environment activated
python -m pip install -r custom_nodes/comfyui-bria-vgl/requirements.txt

# Windows portable: run from ComfyUI_windows_portable
.\python_embeded\python.exe -m pip install -r ComfyUI/custom_nodes/comfyui-bria-vgl/requirements.txt
```

## Connect

1. Add **Bria VGL Prompt**. Type text directly, or right-click the node and convert
   its `text` widget to an input, then connect **Bria VGL Text Input** or another STRING node.
2. Optionally connect the built-in **Load Image** node's IMAGE output to `image`.
3. Select a provider. Leave `model` blank for its default or enter a model override.
4. Connect the `text` output to a text display node or a downstream prompt input.
   Convert the destination text widget to an input if needed. The generator runs
   when a downstream output node is queued.

| Inputs | Behavior |
| --- | --- |
| Text only | Generate a scene prompt |
| Image only (blank text) | Describe the image |
| Text + image | Create the text's requested scene using the image as inspiration |

`VGL JSON` returns the complete structured scene as a STRING, suitable for a
VGL-aware workflow such as FIBO. `Short description` returns the scene's concise
natural-language description for ordinary text prompt inputs. It omits the
detailed structured fields. This node creates prompt text; it does not generate images.

Exactly one image is accepted. Select a single image from a batch upstream.
ComfyUI float images are converted to RGB; transparent pixels are composited on white.
Changing the seed requests a new result; repeatability depends on the provider.
ComfyUI can reuse cached output when inputs are unchanged.

## Providers and credentials

Supports Google AI Studio, Vertex AI, AWS Bedrock, and custom LiteLLM providers,
including OpenAI-compatible endpoints. Set the provider credentials and endpoint
in the environment before launching ComfyUI; the node inherits those settings.

See the [provider setup guide](docs/PROVIDERS.md) for models, environment variables,
Windows and Linux/macOS examples, and troubleshooting.

## Stop and timeout

Click ComfyUI's **Stop/Interrupt** button to cancel a running request. The node
checks for interruption roughly every 0.1 seconds while waiting and terminates
its isolated local VGL worker. It propagates ComfyUI's normal interruption
exception, so canceled results are not returned to downstream nodes.

`timeout_seconds` defaults to 180 seconds and limits the total worker wait,
including imports, provider retries, and fallback attempts. Existing workflows
without this optional input use the same default. A timeout stops the worker and
reports an error. Every new request starts a fresh worker with ComfyUI's Python
and environment; canceled workers are not reused.

Stopping locally does not guarantee that a remote server cancels work it has
already received. The remote request may still finish or incur charges.

### Tests

```powershell
python -m unittest discover -s tests -v
```

Tests use a fake provider; they do not make billable requests. A live provider
smoke test requires your configured credentials and a running ComfyUI installation.
