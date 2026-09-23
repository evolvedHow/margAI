# margai — a portable LLM wrapper framework

A wrapper is a reusable layer of interceptors around any upstream LLM (OpenAI,
OpenRouter, vLLM, Ollama, a local server...). margai gives you the pipeline,
the provider abstraction, and the model routing; **you** write the hooks.

Decorators wrap both sides of an upstream call:

```
client →  @wrapper.request   → provider ⇄ upstream
client ←  @wrapper.response  ← provider   (non-streaming)
client ←  @wrapper.stream    ← provider   (one callback per SSE chunk)
```

Everything observable on a request is OpenAI-compatible dicts, so hooks never
care which provider is behind the call.

## Layout

| Layer | Module | Notes |
| --- | --- | --- |
| Core | `margai.core` | pure stdlib, zero I/O; portable to a Cloudflare Worker |
| Wrapper | `margai.wrapper` | the pipeline + decorators |
| Providers | `margai.providers` | per-upstream normalizers (OpenAI-compatible today) |
| Transport | `margai.transport` | protocol + httpx impl | 
| Adapter | `margai.transport.fastapi` | OpenAI-shaped HTTP API (LibreChat, Modal) |
| Config | `margai.config` | TOML via `tomllib` + env overrides |
| Telemetry | `margai.telemetry` | usage / latency / cost records |

## Quick start

```toml
# margai.toml
[providers.my_openai]
kind = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

[providers.local]
kind = "openai-compatible"
base_url = "http://localhost:11434/v1"
models = ["llama3.2"]
```

```bash
uv run margai            # serves :8000, reads ./margai.toml or $MARGAI_CONFIG
```

Point LibreChat (or any OpenAI SDK) at `http://localhost:8000/v1`. Models show
up namespaced: `marg/my_openai/gpt-4o`, `marg/local/llama3.2`, ... which
populate dropdowns automatically.

## Hooks

```python
from margai import Wrapper

wrapper = Wrapper.from_config(load_config())

@wrapper.request                  # before upstream: inspect/mutate ctx.body, ctx.state
async def enrich(ctx):
    ctx.state["lang"] = detect(ctx.last_user_message())

@wrapper.response                 # after a non-streaming call
def shape(payload, ctx):
    return payload

@wrapper.stream                   # per SSE delta; return None to drop the chunk
def shape_chunks(chunk, ctx):
    return chunk

@wrapper.error                    # shape failures (return a dict to take over)
def on_error(exc, ctx):
    return {"error": {"message": str(exc), ...}}
```

- `request` hooks run in registration order; `response`/`stream`/`error` run
  in reverse (middleware-stack unwind). Use `order=` to shift a hook.
- `ctx.state` threads data from a request hook to a later response/stream hook.
- Hooks are **sync or async** — either is fine.
- Register hooks from config with `[hooks] load = ["myapp.hooks"]`
  (module with `register(wrapper)`) or `["myapp.hooks:register"]`.

## Model routing

`marg/<provider>/<model>` → provider-qualified. `expose = "prefixed" | "both" |
"raw"` controls what `/v1/models` lists. Resolution is offline (configured
`models` lists); upstream discovery is only used for the catalog and is cached.

## Configurability

See `examples/margai.toml.example` for the full surface: prefix, expose mode,
timeouts, per-provider keys/models, telemetry on/off (log, callback, or none),
cost tables, and hook loading. Gateway settings can be overridden with
`MARGAI_HOST`, `MARGAI_PORT`, `MARGAI_PREFIX`, `MARGAI_EXPOSE`, `MARGAI_TIMEOUT`,
`MARGAI_DEFAULT_PROVIDER`.

## Portability

The core never does I/O. Deployments supply a `Transport`:

- **Local / LibreChat / Modal**: `margai.transport.httpx.HttpxTransport` behind
  the FastAPI adapter.
- **Cloudflare Worker**: the same core, minus httpx/FastAPI — implement the
  ~2-method `Transport` protocol over `fetch`.

Running tests: `uv run pytest`.