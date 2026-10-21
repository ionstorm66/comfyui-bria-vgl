# ComfyUI Bria VGL

Create prompts using [Bria-AI/vgl](https://github.com/Bria-AI/vgl).
Includes **Bria VGL Prompt** and **Bria VGL Text Input**, under `Bria/VGL`.
Requires Python 3.10+ and credentials for your selected provider.

For local generation without provider credentials, use **Bria VGL Prepare (Local)**
and **Bria VGL Validate** around ComfyUI's built-in **Generate Text** node.
See the [local/subgraph guide](docs/LOCAL.md) and `examples/local_vgl.json`.

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
| Text + image + mask | Describe an edit to the masked region |

`VGL JSON` returns the complete structured scene as a STRING, suitable for a
VGL-aware workflow such as FIBO. `Short description` returns the scene's concise
natural-language description for ordinary text prompt inputs. It omits the
detailed structured fields. This node creates prompt text; it does not generate images.

Each image input accepts exactly one image. Select a single image from a batch upstream.
ComfyUI float images are converted to RGB; transparent pixels are composited on white.
Changing the seed requests a new result; repeatability depends on the provider.
ComfyUI can reuse cached output when inputs are unchanged.

### Multiple reference images

**Bria VGL Prompt** accepts up to four images. Connect them in order without gaps:
`image` is `<image_1>`, followed by `image_2`, `image_3`, and `image_4`.
With two or more images, enter an edit instruction such as
"put the lamp from <image_2> in <image_1>, using the colors from <image_3>".
The node uses Bria's `EditImages` through `modify()`. Select **VGL JSON** to
retain the structured edit instruction. Different image sizes are supported.

Multiple references require nonempty text and cannot be combined with `mask`;
Bria's masked-edit payload supports one image. Single-image behavior is unchanged.
The `masked_image` output passes through the first image in multi-image mode.
Connect the original references separately to your downstream image model in the
same order; VGL outputs a prompt, not the edited image. All reference images
participate in the prompt cache, so changing any reference requests a new result.

This feature applies to the provider-backed **Bria VGL Prompt** node.
**Bria VGL Prepare (Local)** retains its single-image Generate Text workflow.

### Masked edits

Connect a standard ComfyUI **MASK** to the optional `mask` input, connect the
matching image, and enter an edit instruction such as "replace the chair with
a blue armchair." The node calls VGL's `EditImageWithMask` through `modify()`.

Mask values are converted directly from 0–1 to an 8-bit grayscale PNG:
**1 becomes white (edit)**, **0 becomes black (keep)**, and intermediate values
remain soft grayscale values. No inversion or thresholding is applied.
One mask is accepted as `[1, height, width]` or `[height, width]`. Its dimensions
must match the image; resize it upstream if necessary. A connected mask requires
both an image and nonempty edit text. Disconnect the mask for normal generation.

The second output, **masked_image** (IMAGE), is the complete source image with
the edit region filled with flat **RGB (128, 128, 128)**. It matches the
[FIBO2 pipeline's grey_out preprocessing](https://github.com/muhammadali-afridi/diffusers/blob/ali/fibo2/src/diffusers/pipelines/bria_fibo2/image_processor.py):
the converted 8-bit mask is made binary at **128**. Values at or above 128 replace
the source pixel with gray; lower values retain the converted RGB source pixel.
The original input is not modified. Output dimensions match the source image;
the FIBO2 encoder performs its own subsequent resizing/cropping.

#### Wiring to TextEncodeFibo2

**Feed Bria VGL Prompt's `masked_image` output into TextEncodeFibo2's `image_1` input.**
Use the following connections for masked editing:

| Source | Destination |
| --- | --- |
| Original image (IMAGE) | Bria VGL Prompt → `image` |
| ComfyUI mask (MASK) | Bria VGL Prompt → `mask` |
| Edit instruction text | Bria VGL Prompt → `text` |
| Bria VGL Prompt → `text` (select **VGL JSON**) | TextEncodeFibo2 → `prompt` |
| Bria VGL Prompt → `masked_image` | TextEncodeFibo2 → `image_1` |
| VAE loader → VAE | TextEncodeFibo2 → `vae` |

The original image must be connected to Bria VGL Prompt itself, even if it is
already connected elsewhere in the workflow. Leave `image_2` unused unless
providing another reference image; it is not a mask input. Connect `masked_image`
to **Preview Image** to inspect the gray-filled reference before generation.

Without a mask input, **masked_image** passes through the original input image
unchanged, without quantizing or compositing its pixels. This also applies when
the prompt is served from cache. If no image is connected (text-only generation),
the image output is `None`; leave downstream image nodes disconnected in that case.

This replaces the former **converted_mask** output at the same socket index.
The first output remains **text**. VGL prompting still uses the soft grayscale
mask; thresholding applies only to the FIBO2 reference image output.

VGL uses the mask to mark the requested region in the image shown to the vision
model. The node outputs prompt text and a prepared reference image, not a
generated edit. Use **VGL JSON** to retain the structured edit instruction.
For exact preservation outside the mask, composite the generated result onto
the original with matching dimensions/alignment. A vision-capable provider is required.

### Prompt cache

Each **Bria VGL Prompt** node also keeps its last successful result in memory.
If it executes again with the same text, provider/resolved model, output format,
seed, image content, and mask content, it returns that text without starting an API worker.
Keep the seed's control-after-generate setting on **fixed** to reuse a request.

The image is compared by its converted RGB pixels and dimensions, so reloading
the same image does not require a new request. Changing the seed or another
generation input creates a new request. Changing only the timeout reuses a
successful cached result. Changes to the configured OpenAI base URLs, Vertex
project/location, or AWS profile/region invalidate this node's cache when it runs.

Only successful, uninterrupted results are cached; errors, timeouts, and canceled
requests are not. This cache holds one result per node instance and is cleared
when the instance is recreated or ComfyUI restarts. No cache files are written.
Python integrations can call the node's `clear_cache()` method; in the UI, change
the seed to request a fresh result. ComfyUI's own execution cache remains separate.

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
