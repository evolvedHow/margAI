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

Handlers may be sync or async. Services a call carries (via `ctx.tags`), the
tag-keyed events fire automatically at the matching phase; `before`/`after`/
`stream`/`error` cover both sides of a call plus per-chunk streaming and
failure shaping.

## Marglets

A **marglet** is the unit of extension: one named enhancement, carrying up to
four phase hooks plus a summary, activated per call by a bangtag. The name *is*
the contract — `!margAI: terse` fires every marglet registered as `terse`.

```python
install_bangtags(app)

@app.marglet("terse", summary="Answer in as few words as possible")
def terse(ctx, tag):
    ctx.add_system_prompt("Answer in as few words as possible.")
```

Three equivalent registrations, for the shapes you actually need:

```python
from margAI import Marglet

# 1. decorator, for a single before-hook
@app.marglet("terse", summary="Be brief")
def terse(ctx, tag): ...

# 2. explicit, when the marglet does more than one phase
app.add_marglet(Marglet("terse", before=terse, after=unpad, summary="Be brief"))

# 3. a group, named {marglet}_{phase}
class Writing:
    def terse_before(self, ctx, tag): ...
    def terse_after(self, payload, ctx, tag): ...
app.add_marglet_group(Writing())
```

Inside any phase, `ctx.marglets` lists what the call activated, in dispatch
order. Marglets declare an `order` (lower runs first) and ties break by name,
so execution order does not depend on how the user typed the bangtag.

Registering a name twice raises rather than silently replacing the first
handler; pass `replace=True` when you mean it.

## Dynamic routing

`margAI/dynamic` (configurable via `gateway.dynamic_model`) is a reserved model
id whose target is chosen per call, after every `before` hook has run — which
is what lets a marglet decide the route.

```python
@w.routing.selector(name="prefer-local", order=-10)
def prefer_local(intent, candidates, ctx):
    for c in candidates:
        if c.provider == "local":
            return c          # or a Route, "provider/model", a dict, or a list
    return None               # None abstains; the chain moves on
```

The chain is ordered, and the first non-`None` result wins. Declarative
constraints come from the bangtag and are applied first — `route=`,
`provider=`, `model=`, `not=`, `only=` — and are visible to selectors as the
immutable `intent` (`ctx.intent`, refined in a marglet with `ctx.steer(...)`).

If no selector claims the call it falls back to `gateway.default_provider` and
records `reason="default_provider"`; if that too is excluded by the call's own
constraints, the call fails with a 404 that says so rather than quietly
serving it. Determinism is the point: same config + same tags + same marglets
⇒ same route. `w.routing.describe()` prints the whole chain.

A marglet can own routing for its own calls by passing `select=`:

```python
app.add_marglet(Marglet("go-local", before=..., select=prefer_local, order=-5))
```

Telemetry records `marglets` and `route_reason` on every call, so a `dynamic`
request is never unattributable.

## Layout

| Layer | Module | Notes |
| --- | --- | --- |
| Core | `margAI.core` | pure stdlib, zero I/O; portable to a Cloudflare Worker |
| Wrapper | `margAI.wrapper` | the pipeline + FastAPI-style decorators |
| Marglets | `margAI.core.marglets` | named, bangtag-activated enhancements |
| Routing | `margAI.routing` | the dynamic selector chain |
| Bangtags | `margAI.bangtag` | optional `!margAI:` directive layer (`install_bangtags`) |
| Providers | `margAI.providers` | per-upstream normalizers (OpenAI-compatible, Anthropic) |
| Transport | `margAI.transport` | protocol + httpx impl | 
| Adapter | `margAI.transport.fastapi` | OpenAI-shaped HTTP API (LibreChat, Modal) |
| Config | `margAI.config` | TOML via `tomllib` + env overrides |
| Telemetry | `margAI.telemetry` | usage / latency / cost records |

## Quick start

```toml
# margAI.toml
[gateway]
default_provider = "my_openai"

[providers.my_openai]
kind = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
models = ["gpt-4o", "gpt-4o-mini"]

[providers.local]
kind = "openai-compatible"
base_url = "http://127.0.0.1:8080/v1"
default_model = "qwen2.5-coder"
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
up namespaced: `margAI/my_openai/gpt-4o`, `margAI/local/qwen2.5-coder`, ...
which populate dropdowns automatically. Serving: chat, legacy completions,
embeddings, image generation, audio transcription, and models — see
`docs/ENDPOINTS.md`.

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
is what the tag-keyed events dispatch on. The namespace is required, so
user-typed text can't accidentally fire a handler.

Notes:

- `before` hooks run in registration order; `after`/`stream`/`error` run in
  reverse (middleware-stack unwind). Use `order=` to shift a hook.
- `ctx.state` threads data from a request hook to a later response/stream hook.
- Hooks/events are **sync or async** — either is fine.
- Errors never leak internals: an unhandled exception returns a generic 500 to
  the client, with the detail in the log.
- Register a module from config with `[hooks] load = ["myapp.hooks"]`
  (module with `register(wrapper)`) or `["myapp.hooks:register"]`.

See `src/margAI/examples/` for `simple_marglets.py` (all three registration
forms) and `optimize_hooks.py` (a marglet using every phase).

## Model routing

`margAI/<provider>/<model>` → provider-qualified. A bare model id resolves
against every provider's configured `models` list. `expose = "prefixed" |
"both" | "raw"` controls what `/v1/models` **lists** — it never gates what the
router **accepts**; conflating the two used to make `expose = "prefixed"`
silently reject every bare id. Resolution is offline (configured `models`
lists); upstream discovery is only used for the catalog and is cached (TTL from
`models_cache_ttl`). Call `wrapper.invalidate_models_cache()` to drop the
catalog early.

## Configurability

See `examples/margAI.toml.example` for the full surface: prefix, expose mode,
timeouts, per-provider keys/models, telemetry on/off (log, callback, or none),
cost tables, and hook loading. Gateway settings can be overridden with
`MARGAI_HOST`, `MARGAI_PORT`, `MARGAI_PREFIX`, `MARGAI_EXPOSE`, `MARGAI_TIMEOUT`,
`MARGAI_DEFAULT_PROVIDER`, `MARGAI_DYNAMIC_MODEL`.

## Portability

The core never does I/O. Deployments supply a `Transport`:

- **Local / LibreChat / Modal**: `margAI.transport.httpx.HttpxTransport` behind
  the FastAPI adapter.
- **Cloudflare Worker**: the same core, minus httpx/FastAPI — implement the
  ~2-method `Transport` protocol over `fetch`.

Running tests: `uv run pytest`.