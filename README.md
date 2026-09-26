# margAI — a portable LLM wrapper framework

A wrapper is a reusable layer of interceptors around any upstream LLM (OpenAI,
OpenRouter, vLLM, Ollama, a local server...). margAI gives you the pipeline,
the provider abstraction, and the model routing; **you** write the handlers.
The developer experience is FastAPI-flavored: decorate handlers straight onto
your wrapper.

```python
from margAI import Wrapper, install_bangtags
from margAI.config import load_config

app = Wrapper.from_config(load_config())

install_bangtags(app)                 # layer-in "!margAI: <tag>" parsing

@app.before("refine")                 # request-phase event (tag-keyed)
def refine(ctx, tag):
    ctx.state["refined"] = compact(ctx.last_user_message())

@app.after("structure")               # response-phase event
def structure(payload, ctx, tag):
    payload["choices"][0]["message"]["content"] = bulletize(
        payload["choices"][0]["message"]["content"])
    return payload

@app.after                           # per-call hooks (every call)
def annotate(payload, ctx):
    return payload
```

Handlers may be sync or async. Services a call carries (via
`ctx.state["bangtags"]`), the tag-keyed events fire automatically at the
matching phase; `before`/`after`/`stream`/`error` cover both sides of a call
plus per-chunk streaming and failure shaping.

## Layout

| Layer | Module | Notes |
| --- | --- | --- |
| Core | `margAI.core` | pure stdlib, zero I/O; portable to a Cloudflare Worker |
| Wrapper | `margAI.wrapper` | the pipeline + FastAPI-style decorators |
| Bangtags | `margAI.bangtag` | optional `!margAI:` directive layer (`install_bangtags`) |
| Providers | `margAI.providers` | per-upstream normalizers (OpenAI-compatible, Anthropic) |
| Transport | `margAI.transport` | protocol + httpx impl | 
| Adapter | `margAI.transport.fastapi` | OpenAI-shaped HTTP API (LibreChat, Modal) |
| Config | `margAI.config` | TOML via `tomllib` + env overrides |
| Telemetry | `margAI.telemetry` | usage / latency / cost records |

## Quick start

```toml
# margAI.toml
[providers.my_openai]
kind = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

[providers.local]
kind = "openai-compatible"
base_url = "http://127.0.0.1:8080/v1"
models = ["qwen2.5-coder"]

# [providers.anthropic]
# kind = "anthropic"
# base_url = "https://api.anthropic.com/v1"
# api_key_env = "ANTHROPIC_API_KEY"
# models = ["claude-sonnet-4-5"]
```

```bash
uv run margAI            # serves :8000, reads ./margAI.toml or $MARGAI_CONFIG
```

Point LibreChat (or any OpenAI SDK) at `http://localhost:8000/v1`. Models show
up namespaced: `margAI/my_openai/gpt-4o`, `margAI/local/llama3.2`, ... which
populate dropdowns automatically.

Both `POST /v1/chat/completions` and the legacy `POST /v1/completions` surface
are served (streaming included).

## Handlers and events

Every decorator has a tag-keyed and a per-call form. `name` (a string first
argument, or `tag=`) selects the tag-keyed form; no name means "every call".

```python
from margAI import Wrapper, EventHandler

wrapper = Wrapper.from_config(load_config())

@wrapper.before("route")                  # tag-keyed event: fn(ctx, tag)
def route(ctx, tag): ...

@wrapper.after                            # per-call hook: fn(payload, ctx) -> payload
def shape(payload, ctx): return payload

@wrapper.stream                           # per SSE delta; return None to drop it
def shape_chunks(chunk, ctx): return chunk

@wrapper.error                            # shape failures (return a dict to take over)
def on_error(exc, ctx): return None
```

Events are stored on `wrapper.events` (`all()` / `describe()`); a group can
also be written as an `EventHandler` subclass with methods named
`{event}_{tag}` and attached once with `wrapper.add_handler(MyGroup())`.

The bangtag layer (`margAI.bangtag`) makes tags user-facing: `install_bangtags`
parses `!margAI: refine, route=...` out of the last user message, strips it
from the prompt, and stores `Tag` objects in `ctx.state["bangtags"]` — which
is what the tag-keyed events dispatch on.

Notes:

- `before` hooks run in registration order; `after`/
  `stream`/`error` run in reverse (middleware-stack unwind). Use `order=` to
  shift a hook.
- `ctx.state` threads data from a request hook to a later response/stream hook.
- Hooks/events are **sync or async** — either is fine.
- Register a module from config with `[hooks] load = ["myapp.hooks"]`
  (module with `register(wrapper)`) or `["myapp.hooks:register"]`.

See `margAI-devkit/examples/customize_me.py` for a complete pass-through
scaffold of every decorator with customization notes.

## Model routing

`margAI/<provider>/<model>` → provider-qualified. `expose = "prefixed" | "both" |
"raw"` controls what `/v1/models` lists. Resolution is offline (configured
`models` lists); upstream discovery is only used for the catalog and is cached
(TTL from `models_cache_ttl`). Call `wrapper.invalidate_models_cache()` to drop
the catalog early.

## Configurability

See `examples/margAI.toml.example` for the full surface: prefix, expose mode,
timeouts, per-provider keys/models, telemetry on/off (log, callback, or none),
cost tables, and hook loading. Gateway settings can be overridden with
`MARGAI_HOST`, `MARGAI_PORT`, `MARGAI_PREFIX`, `MARGAI_EXPOSE`, `MARGAI_TIMEOUT`,
`MARGAI_DEFAULT_PROVIDER`.

## Portability

The core never does I/O. Deployments supply a `Transport`:

- **Local / LibreChat / Modal**: `margAI.transport.httpx.HttpxTransport` behind
  the FastAPI adapter.
- **Cloudflare Worker**: the same core, minus httpx/FastAPI — implement the
  ~2-method `Transport` protocol over `fetch`.

Running tests: `uv run pytest`.