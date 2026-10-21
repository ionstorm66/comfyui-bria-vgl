# Providers and credentials

Set credentials in the environment **before launching ComfyUI**. Credentials are
not stored in node widgets or workflow files. Requests send the input text and,
when connected, the image to the selected provider and may incur API charges.

| Provider | Default model | Environment |
| --- | --- | --- |
| Google AI Studio | `gemini/gemini-3.5-flash-lite` | `GEMINI_API_KEY` |
| Google Vertex AI | `vertex_ai/gemini-3.5-flash-lite` | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`, `GOOGLE_APPLICATION_CREDENTIALS` (service account file path) |
| AWS Bedrock | `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Standard AWS credentials/profile and `AWS_REGION_NAME` |
| Custom (LiteLLM) | Required | Provider-specific environment variables |

For built-in providers, model overrides may omit the provider prefix. Custom
requires a full LiteLLM model ID, for example `openai/<your-model>`; configure
that provider's credentials separately. Custom model compatibility is determined
by VGL/LiteLLM and the model's structured-output and image capabilities. Bria
tunes VGL's prompts for Gemini. A Bria image-generation API key is not used here.

## Custom OpenAI-compatible endpoints

For a local model server, proxy, or hosted service that implements the
OpenAI-compatible Chat Completions API, select **Custom (LiteLLM)** and use
`openai/<model-id>` in the node's **model** field. Replace `<model-id>` with the
exact model ID exposed by your server, not its URL or a friendly display name.

Configure the endpoint in the environment before starting ComfyUI:

| Variable | Value |
| --- | --- |
| `OPENAI_API_BASE` | Your API base URL, for example `http://localhost:1234/v1` |
| `OPENAI_BASE_URL` | Alternative recognized by LiteLLM; set it to the same URL to avoid a conflicting existing value |
| `OPENAI_API_KEY` | The key issued by your endpoint, or a nonempty placeholder only if your local server requires no authentication |

Use the base URL supplied by your provider, typically ending in `/v1`.
Do not append `/chat/completions`; the client adds that path. A website's chat
page URL is not an API endpoint.

### Windows PowerShell: current session

Replace the sample URL and key, then launch ComfyUI from **that same PowerShell
window**. These settings apply to that shell and programs launched from it:

```powershell
$env:OPENAI_API_BASE = "http://localhost:1234/v1"
$env:OPENAI_BASE_URL = $env:OPENAI_API_BASE
$env:OPENAI_API_KEY = "your-endpoint-key"

# Standard install: activate your usual ComfyUI environment first.
Set-Location "C:\ComfyUI"
python main.py
```

For Windows portable, use the same environment assignments above, then launch
your usual portable batch file instead:

```powershell
Set-Location "C:\ComfyUI_windows_portable"
.\run_nvidia_gpu.bat
```

Replace these example installation paths with your own. Use your usual launcher
if it differs. For a local server with authentication disabled, a value such as
`"local-not-required"` satisfies the client's nonempty-key requirement; it does
not bypass authentication on a server that requires a real key.

In **Bria VGL Prompt**, set:

- **provider:** `Custom (LiteLLM)`
- **model:** `openai/your-server-model-id`
- **output_format:** `VGL JSON` or `Short description`

The node's worker inherits the ComfyUI process environment, including these
variables. Changing a separate terminal's variables does not update a running
ComfyUI process. Stop ComfyUI and relaunch it from the configured terminal.
All nodes using this OpenAI-compatible environment share the same endpoint;
the node does not currently provide a separate URL/key field per node.

### Windows: persist settings for future sessions

To save the endpoint and key in your Windows user environment:

```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_BASE", "http://localhost:1234/v1", "User")
[Environment]::SetEnvironmentVariable("OPENAI_BASE_URL", "http://localhost:1234/v1", "User")
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your-endpoint-key", "User")
```

These commands do not update an already-running terminal or ComfyUI. Open a new
terminal after saving them, then start ComfyUI. If your terminal or launcher
still inherits old values from its parent application, restart that application
or use the current-session assignments above. User environment variables are
persistent configuration, not an encrypted secret store.

### Linux/macOS

```bash
export OPENAI_API_BASE="http://localhost:1234/v1"
export OPENAI_BASE_URL="$OPENAI_API_BASE"
export OPENAI_API_KEY="your-endpoint-key"
# From your ComfyUI folder, with its Python environment activated:
python main.py
```

### Other LiteLLM providers

The Custom option also accepts provider-specific LiteLLM model IDs. Use that
provider's documented prefix and credential variables; the `OPENAI_*` example
above applies to the `openai/` route, not every provider. See
[LiteLLM providers](https://docs.litellm.ai/docs/providers) and
[OpenAI-compatible endpoints](https://docs.litellm.ai/docs/providers/openai_compatible).

### Troubleshooting

- **Connection refused / timeout:** confirm the model server is running and the
  URL is reachable from the machine running ComfyUI. In a container, `localhost`
  refers to the container, not automatically the host machine.
- **404:** check the provider's API base path (often `/v1`) and the exact model ID.
- **401 / 403:** check the endpoint key and permissions.
- **Requests go to the wrong URL:** check both base-URL variables. Current LiteLLM
  prefers `OPENAI_BASE_URL` over `OPENAI_API_BASE`; the examples set both equally.
- **Invalid VGL / unsupported response format:** the endpoint must accept VGL's
  JSON response-format request and its model must produce valid VGL. General
  OpenAI API compatibility alone does not guarantee this.
- **Image requests fail:** use a vision-capable model and endpoint. Text-only
  models can still be used with the image input disconnected.

[Back to README](../README.md)
