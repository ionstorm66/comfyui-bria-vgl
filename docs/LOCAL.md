# Local VGL with Generate Text

Restart ComfyUI after updating this package. Add **Bria VGL Prepare (Local)**,
ComfyUI's built-in **Generate Text**, and **Bria VGL Validate**.
The existing bria-vgl dependency provides the templates and validation model;
these nodes do not call a provider or require API credentials.

| Source | Destination |
| --- | --- |
| Supported generative model loader → CLIP | Generate Text → clip |
| Prepare → system_prompt | Generate Text → system_prompt |
| Prepare → prompt | Generate Text → prompt |
| Prepare → image (when using an image) | Generate Text → image |
| Generate Text → generated_text | Validate → generated_text |
| Prepare → require_edit_instruction | Validate → require_edit_instruction (convert widget to input) |
| Validate → vgl_json | TextEncodeFibo2 → prompt |
| Prepare → masked_image (for masked editing) | TextEncodeFibo2 → image_1 |

Keep Generate Text's **use_default_template enabled**: its system_prompt input
is ignored when that option is disabled. Start with **thinking off**,
**max_length 7000**, and sampling on with **temperature 0.2** and a fixed seed.
These are starting settings, not a guarantee of valid or high-quality output.
Use a supported generative CLIP model; conventional CLIP encoders cannot generate
text. Image inputs require a vision-capable model. Update ComfyUI if Generate Text
does not expose system_prompt.

Enter your scene in Prepare's text widget. Connect an image for inspiration, or
leave text blank to caption it. For masked editing, connect the original image,
matching mask, and a nonempty edit instruction to Prepare. Its image output uses
VGL's soft gray mask for vision. Its separate masked_image output uses FIBO's
binary gray-out preprocessing. Without a mask, masked_image passes through the
original image. With text only, leave both image outputs disconnected.

Select the three nodes and use ComfyUI's Convert to Subgraph action. Expose text,
CLIP, optional image/mask, and generation settings as inputs, and vgl_json,
short_description, and masked_image as outputs. The supplied
`examples/local_vgl.json` is an ungrouped workflow starting point; connect your
model loader and downstream FIBO node after importing it.

Validation accepts raw JSON or one enclosing JSON Markdown fence. It uses strict
ThalesVGL type validation and requires a nonempty short description; masked edits
also require a nonempty edit_instruction. Unknown fields are dropped by Bria's
model. Validation does not assess visual accuracy, enforce every natural-language
field guideline, or retry generation. On failure, increase max_length if output
was truncated, adjust the prompt/model, or change the seed and queue again.

Generation and model memory are managed by ComfyUI. The API worker's timeout and
one-result cache do not apply to this path; normal ComfyUI execution caching does.
