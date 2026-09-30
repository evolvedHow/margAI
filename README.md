# margAI — a portable LLM wrapper framework

A wrapper is a reusable layer of interceptors around any upstream LLM (OpenAI,
OpenRouter, vLLM, Ollama, a local server...). margAI gives you the pipeline,
the provider abstraction, and the model routing; **you** write the handlers.

## Quick Start (Simplified Gateway API)

The simplest way to use margAI is with the Gateway class:

```python
from margAI import Gateway

app = Gateway()  # Reads ./margAI.toml automatically

@app.tag("summarize")
def summarize(ctx, tag):
    """Add instruction to summarize."""
    ctx.add_system("Provide a brief summary.")

@app.tag("rag")
def add_context(ctx, tag):
    """Look up context from vector DB."""
    query = tag.value or ctx.user_message()
    docs = vector_db.search(query, top_k=5)
    ctx.add_system(f"Relevant context:\n{docs}")

@app.tag("fast")
def route_fast(ctx, tag):
    """Route to a fast local model."""
    return ctx.route_to("margAI/local/llama3.2-1b")

app.serve()
```

Users can then write prompts like:
```
!app: rag=quantum-computing, summarize

What is quantum entanglement?
```

A directive is one line: `!<namespace>: tag, tag=value, ...`. Tags are split on
commas and then on whitespace, so **a value cannot contain a space** — use `-`
or `_`, or pass structured data some other way. Quoting is not a thing:
`rag="quantum computing"` parses as the value `"quantum` plus a junk `computing"`
tag.

The model id is whatever `GET /v1/models` returned — under the default
`prefixed` expose that means `<namespace>/<provider>/<model>`. A bare
`<namespace>/dynamic` hands each call to the routing chain instead.

**Key principles:**
- **One decorator**: `@app.tag()` for everything a caller can ask for
- **Built-ins included**: `route=`, `provider=`, `model=`, `not=`, `only=`,
  `cost=`, `think=` work with no setup, and are disabled with
  `[packs."margAI.builtin_tags"] enabled = false`
- **Complete flexibility**: routing, caching, and validation are all just tags
- **Framework owns infrastructure**: pipeline, telemetry, exit points
- **You own the logic**: when to hook in, and what to do

See `examples/minimal_gateway.py` and `examples/rag_gateway.py` for working
examples.

## Two APIs, One Wrapper

`Gateway` is a decorator-first front end over the same `Wrapper`. It reads
`./margAI.toml`, parses bangtags for your namespace, and exposes `@app.tag()`.
Everything else is available on `app.wrapper` when you outgrow it.

```python
from margAI import Gateway

app = Gateway()  # ./margAI.toml, or $MARGAI_CONFIG

@app.tag("summarize")
def summarize(ctx, tag):
    ctx.add_system("Provide a brief summary.")

app.serve()
```

## Advanced API (Full Wrapper)

For more control, use the full Wrapper API:

```python
from margAI import Wrapper
from margAI.config import load_config

app = Wrapper.from_config(load_config())

# Bangtag parsing is already installed for the reserved `margAI` namespace,
# along with the built-in `route=` / `provider=` / `model=` / `think=` tags.
# Add a second namespace only if you renamed yourself:
from margAI import install_bangtags
install_bangtags(app, namespace="myapp")

@app.before("refine")  # Tag-keyed hook
def refine(ctx, tag):
    ctx.add_system_prompt("Refine your response.")

@app.after  # Runs on every response
def log_response(payload, ctx):
    print(f"Tokens: {payload.get('usage', {}).get('total_tokens')}")
    return payload
```

Handlers may be sync or async. `before`/`after`/`stream`/`error` cover both
sides of a call plus per-chunk streaming and failure shaping.

Built-in tags are installed by `Wrapper` itself, so `Gateway` and `Wrapper`
behave identically here. To turn them off:

```toml
[packs."margAI.builtin_tags"]
enabled = false
```

## Concepts

margAI has three layers, each building on the previous:

1. **Tags** - Handlers that run when a bangtag mentions them
   - `@app.tag("summarize")` on `Gateway`, or `app.on("summarize")` on
     `Wrapper` → runs when the user writes `!app: summarize`
   - Can modify prompts, override routing, short-circuit, or return errors

2. **Hooks** - Functions that run on every request/response
   - `@app.before` / `@app.on_request` → runs before every call
   - `@app.after` / `@app.on_response` → runs after every call
   - Useful for logging, metrics, caching

3. **Routing** - Just another tag that returns `ctx.route_to(model)`
   - `!<namespace>/dynamic` defers to a selector chain you build with
     `wrapper.routing.add_selector(...)`
   - Or a `[models.*]` virtual model, which picks a target per call by policy

Most apps only need layer 1 (tags).

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

A `route=` that pins both a provider and a model short-circuits the chain:
`!margAI: route=local/qwen3` asked for a specific target. The pin is still
subject to the call's own constraints — `!margAI: route=openai/x, not=openai`
is a contradiction and fails with a 404 that says so, rather than quietly
serving openai. Same rule as the fallback below.

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
| Bangtags | `margAI.bangtag` | the `!margAI:` directive layer, installed by default |
| Built-ins | `margAI.builtin_tags` | `route=`, `provider=`, `model=`, `not=`, `only=`, `cost=`, `think=` |
| Providers | `margAI.providers` | per-upstream normalizers (OpenAI-compatible, Anthropic) |
| Transport | `margAI.transport` | protocol + httpx impl | 
| Adapter | `margAI.transport.fastapi` | OpenAI-shaped HTTP API (LibreChat, Modal) |
| Config | `margAI.config` | TOML via `tomllib` + env overrides |
| Telemetry | `margAI.telemetry` | usage / latency / cost records |

## Complete Example: RAG + Caching + Smart Routing

```python
from margAI import Gateway
import chromadb

app = Gateway()
db = chromadb.Client()
cache = {}

@app.tag("rag")
def add_rag_context(ctx, tag):
    """Add vector DB results to prompt."""
    query = tag.value or ctx.user_message()
    results = db.search(query, n_results=5)
    ctx.add_system(f"Context:\n{results}")

@app.tag("cache")
def check_cache(ctx, tag):
    """Return cached response if available."""
    key = ctx.cache_key()
    if key in cache:
        return ctx.respond(cache[key])  # Skip LLM

@app.after
def store_cache(payload, ctx):
    """Cache responses."""
    key = ctx.cache_key()
    cache[key] = payload
    return payload

@app.tag("smart")
def smart_routing(ctx, tag):
    """Route based on complexity."""
    if ctx.analyze_complexity() == "high":
        return ctx.route_to("margAI/openai/gpt-4o")
    else:
        return ctx.route_to("margAI/local/llama3.2-1b")

app.serve()
```

**Usage:**
```
!app: rag=quantum-computing, cache, smart

Explain quantum entanglement
```

This will:
1. Check cache (skip if hit)
2. Search vector DB for "quantum computing"
3. Analyze complexity and route appropriately
4. Cache the response

**Framework owns**: Pipeline, telemetry, provider abstraction  
**You own**: When to cache, where to search, how to route

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
from the prompt, and records `Tag` objects with `ctx.set_tags(...)` — which is
what the tag-keyed events dispatch on. The namespace is required, so
user-typed text can't accidentally fire a handler. `install_bangtags` is
optional: if you don't want `!margAI:` syntax, skip it and call
`ctx.set_tags(my_tags_for(ctx))` from your own request hook instead — at
`order=-100`, since the tag dispatch runs at `-50`.

Because a directive is a routing instruction parsed out of user text, an
allowlist is the trust boundary when the user is not you — see
[Security](#security).

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

## Virtual models

`margAI/fast` is not a model anyone hosts. It is a *claim* about one, resolved
per call against the models actually configured — so the client contract stays
stable while the catalogue underneath it changes.

```toml
[models.fast]
strategy = "cheapest"          # first | cheapest | round_robin | least_used | highest_balance
providers = ["openai", "gemini"]
exclude_providers = ["openrouter"]
prefer = ["openai/gpt-4o-mini"]   # tried first; a miss is skipped, not fatal
max_cost_per_1m = 1.0          # blended $/1M, needs [telemetry.costs]
tags = ["chat"]                # eligible only when the call carries these
```

`cheapest` needs prices in `[telemetry.costs]`; without them every model is
equally priceless and the cheapest strategy degrades to `first`. `round_robin`
keeps one cursor per policy behind a lock, so it spreads load across threads
rather than handing every request the same first target.

`least_used` picks the candidate with the fewest calls this billing window.
`highest_balance` picks the one with the most budget left, so a cap you
declared is spent down rather than exhausted. Defining budgets is optional, and
a candidate with **no budget is immaterial**: it is ranked as a neutral zero,
neither rewarded for an unspendable cap nor punished for an unmeasurable one.

Both read an in-process ledger, fed with the same estimated cost telemetry
reports. That ledger is per-process and never persisted (a telemetry log on
disk records the same figures, but nothing reads them back), so under multiple
uvicorn workers each process counts only its own traffic — the same caveat as
`round_robin`'s cursor. `highest_balance` is a way of rationing a budget *you
declared*, not a way of reading a provider's account: margAI never calls a
billing API.

Metering — and so `highest_balance` — needs **both** prices in
`[telemetry.costs]` and a cycle (`reset_day`, or an explicit period; see the
example config). With either missing it degrades to `least_used` rather than
ranking on a number it cannot stand behind. Budgets key like costs:
`"provider/model"`, a bare `"model"` for any provider, or `"provider/*"` for a
whole provider.

A virtual name that collides with a real provider or model is rejected at
construction, not resolved by whichever code path ran first.

A caller can override a policy per call through `[models.*]` in the per-call
config below. The override is a *narrowing* the caller's own `not=`/`only=`
constraints still apply on top of.

## Per-call config

`wrapper.complete(body, config=...)` and `open_stream` take a config overlay
as TOML text or a dict. Handlers read it through `ctx.config`:

```python
@app.before
def guard(ctx):
    if ctx.config.enabled("audit") and ctx.config.pack("audit", "threshold"):
        ...
```

The surface is deliberately narrow. Only `[packs.*]` and `[models.*]` are
accepted — never hooks, providers, callbacks, imports or endpoints, because an
overlay is untrusted input arriving over HTTP. Tables deep-merge, lists
replace, and anything else is a `400` *before* any handler runs, so a rejected
config cannot have had a side effect.

## Security

**margAI does not authenticate.** The FastAPI surface accepts any request that
reaches it, and the per-call overlay guards above limit *what a caller may
configure* — they protect nothing from a caller who was never allowed in the
first place. Treat the gateway as an internal component:

- Bind to loopback. `MARGAI_HOST=127.0.0.1` (or `host = "127.0.0.1"` in
  `[gateway]`) is the default you want on a single host. The shipped default
  is `0.0.0.0`, which is a convenience for containers, not a recommendation.
- To expose it, put an authenticating reverse proxy in front — Caddy, nginx,
  an ingress with an auth annotation — and let the proxy handle TLS,
  authentication and rate limiting. margAI will not do it for you.
- In a container, publish the port to `127.0.0.1` on the host
  (`127.0.0.1:8000:8000`), not to `0.0.0.0`.

Anyone who can send a request can spend your provider budget by naming an
expensive model in the `model` field, so authentication is the control that
matters. Everything below is defense in depth behind it.

**Bangtags are a control channel inside the user channel.** `!margAI:`
directives are parsed out of the last user message, so `route=openai/gpt-4o` in
a prompt is a routing instruction written by whoever wrote the prompt. If that
person is your own application, `install_bangtags(app)` is correct as-is. If it
is a user you do not control, pass an allowlist:

```python
install_bangtags(app, allow=["think", "terse", "refine"])
```

Anything else on a directive is dropped with a warning, and the directive text
is stripped from the prompt either way — denying a tag must not leave
`!margAI: route=...` sitting in the model's context for an
instruction-following model to act on. The allowlist can be set at any point,
including after the built-in pack has already installed the `margAI` namespace,
and tightening is one-way. An allowlist does not replace authentication: a
caller who can send a request can still set the `model` field directly.

**Secrets belong in the environment.** `ProviderConfig` takes `api_key_env`
(a variable name) rather than a literal `api_key`, and `.env` is gitignored —
keep it that way. The embedded default config ships provider shells with no
keys, no hook auto-loading and no localhost upstreams.

## Packs

Any installable package can extend the gateway by advertising a `margAI.packs`
entry point:

```toml
[project.entry-points."margAI.packs"]
my_pack = "my_pack:install"
```

Each discovered pack runs and registers tags in its own namespace. A pack that
fails to import is recorded in `wrapper.pack_failures` and skipped rather than
taking the gateway down with it — except one named in
`[gateway] packs_only`, where failing is the point. Set
`[gateway] pack_discovery = false` to skip discovery entirely.

## Diagnostics

```console
$ margAI doctor
  ok    config: margAI.toml over embedded
  WARN  provider openai: no API key (expected $OPENAI_API_KEY)
        Requests will fail upstream until it is set.
  note  routing: no selectors registered, so 'margAI/dynamic' can only fall back
        Add wrapper.routing.add_selector(...) or set gateway.default_provider.
```

`margAI doctor` checks config layering, pack failures, undocumented tags,
provider models and keys, virtual-model ceilings, and whether anything can
actually route. It exits non-zero on failures, `--json` for CI, and it never
sends a request — it diagnoses wiring, and a diagnostic with its own side
effects is one nobody trusts.

`GET /v1/margAI/tags` serves the same live picture as `wrapper.describe()`:
registered tags by namespace, configured virtual models, providers, packs, and
config provenance.

Per-call telemetry goes wherever `[telemetry] emit` points: the process log
(`log`), your callback (`callback`), or a generated JSONL file (`file`). The
file sink writes one JSON object per line to
`<label>_telemetry_<yymmddhhmmss>.log`, where the label defaults to the gateway
prefix and the directory to `[telemetry] dir` or `$XDG_STATE_HOME/margAI/logs`.
The timestamp in the name is the run's start, so restarts append to new files
rather than clobbering the log you were reading.

## Configurability

See `examples/margAI.toml.example` for the full surface: prefix, expose mode,
timeouts, per-provider keys/models, telemetry on/off (log, callback, file, or
none), cost tables, billing cycle and budgets for `highest_balance`, and hook
loading.
Gateway settings can be overridden with
`MARGAI_HOST`, `MARGAI_PORT`, `MARGAI_PREFIX`, `MARGAI_EXPOSE`, `MARGAI_TIMEOUT`,
`MARGAI_DEFAULT_PROVIDER`, `MARGAI_DYNAMIC_MODEL`.

Configuration is layered, lowest to highest: dataclass defaults, the embedded
`src/margAI/_default.toml`, your `margAI.toml` (or `-c`/`$MARGAI_CONFIG`),
`MARGAI_*` environment variables, and a per-call overlay. The embedded layer
carries provider shells only — no hook auto-loading, no localhost upstreams, no
literal API keys — so a fresh install is configured but not yet pointed
anywhere. `config.layers` records which layers actually applied.

Naming a config file that does not exist (`-c` or `$MARGAI_CONFIG`) is an
error. A missing `./margAI.toml` is not: that is the normal no-config case,
and silently starting a gateway with none of the operator's providers is worse
than saying so.

## Portability

The core never does I/O. Deployments supply a `Transport`:

- **Local / LibreChat / Modal**: `margAI.transport.httpx.HttpxTransport` behind
  the FastAPI adapter.
- **Cloudflare Worker**: the same core, minus httpx/FastAPI — implement the
  ~2-method `Transport` protocol over `fetch`.

Running tests: `uv run pytest`.

Linting and type checks (both configured in `pyproject.toml`):

```sh
uv run ruff check src tests
uv run mypy src/margAI
```

`mypy` is configured over `src/margAI` only. The test suite is deliberately not
annotated end to end, so running it over `tests` reports a wall of
`no-untyped-def` that says nothing about the library.

## License

[Apache-2.0](LICENSE) © 2026 Vish Ganapathy. Free to use commercially and
closed-source, including if you embed it in a product you never open up.

What that obliges you to do, in full:

- **Ship the license.** Anyone you distribute the Work to gets a copy of
  Apache-2.0.
- **Mark your changes.** Files you modified must say so.
- **Keep the attribution.** The `NOTICE` file's contents must travel with any
  derivative work, in a `NOTICE` file, in your source or docs, or in a display
  the user actually sees. This is section 4(d), and it is the part that makes
  "credit margAI" a real requirement rather than a suggestion.

There is no copyleft and no network clause. Embedding margAI in a commercial
product does not oblige you to release any of your own source, and your own
licensing choices are unaffected.

And the ask, which the license does not enforce: **if margAI is load-bearing in
something you built, say so.** A "built with margAI" line in your README, an
About page, or a blog post costs you nothing and is the only credit I can
actually ask for. It is a request, not a term — §4(d) above is the part with
teeth.

Contributions are accepted under the same terms; see
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Security reports go to
[SECURITY.md](SECURITY.md), not the issue tracker.
