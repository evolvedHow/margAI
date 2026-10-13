# marga

**मार्ग (mārga)** - Sanskrit for "path" or "route"

**Ultra-minimal LLM routing proxy** - a transparent passthrough layer with smart routing.

## What it does

- **OpenAI-compatible API** - drop-in replacement for any OpenAI client
- **Smart routing** - usage-based routing to the cheapest/least-used backend
- **Prompt hooks** - inject context before/after LLM (perfect for RAG)
- **Zero overhead** - <50ms import, minimal latency

## Size

- ~500 lines total (vs 10,389 in full margAI)
- 2 dependencies: `httpx`, `fastapi`
- <50ms cold import time (vs 380ms)

## Usage

```python
from marga import Proxy

# Configure backends
proxy = Proxy(backends={
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-...",
        "models": ["gpt-4o", "gpt-4o-mini"]
    },
    "local": {
        "base_url": "http://127.0.0.1:8080/v1",
        "models": ["qwen2.5-coder"]
    }
})

# Add routing logic
@proxy.route
def smart_route(ctx):
    """Route to cheapest available backend."""
    if ctx.estimated_tokens < 1000:
        return "local/qwen2.5-coder"
    return "openai/gpt-4o-mini"

# Add prompt hooks
@proxy.before
def add_rag_context(ctx):
    """Inject RAG context before LLM."""
    ctx.messages.insert(0, {
        "role": "system",
        "content": "Relevant context: ..."
    })

@proxy.after
def log_response(ctx, response):
    """Process response after LLM."""
    print(f"Tokens used: {response['usage']['total_tokens']}")
    return response

# Start server
proxy.serve(host="127.0.0.1", port=8000)
```

## Architecture

```
Client Request (OpenAI format)
    ↓
[Before Hooks] ← Prompt injection, RAG, etc.
    ↓
[Router] ← Usage tracking, cost optimization
    ↓
[Backend] ← Actual LLM (OpenAI/local/etc)
    ↓
[After Hooks] ← Response processing
    ↓
Client Response
```

## Differences from margAI (the full framework)

**Removed:**
- Bangtag parsing (433 lines)
- Virtual models (617 lines)
- Complex config system (907 lines)
- Diagnostics (344 lines)
- Packs/extensions (253 lines)
- Billing/telemetry (245 lines)

**Kept:**
- Routing engine
- Before/after hooks
- OpenAI-compatible API
- HTTP passthrough

## Performance

- Import time: <50ms (vs 380ms)
- Request overhead: <5ms
- Memory: ~10MB (vs ~50MB)
