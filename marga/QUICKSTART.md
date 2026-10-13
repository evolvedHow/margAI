# marga Quick Start

**मार्ग (mārga)** - Sanskrit for "path" or "route"

Get your LLM routing proxy running in 60 seconds.

## Install

```bash
cd marga
uv venv
source .venv/bin/activate  # or `.venv/bin/activate.fish` for fish shell
uv pip install -e .
```

## Minimal Example (5 lines)

```python
from marga import Proxy

proxy = Proxy(backends={
    "local": {"base_url": "http://127.0.0.1:8080/v1", "models": ["qwen2.5-coder"]}
})

proxy.serve()  # Starts on http://127.0.0.1:8000
```

**That's it!** Point any OpenAI client at `http://127.0.0.1:8000/v1`.

## Complete Example (with routing + RAG)

Save as `my_proxy.py`:

```python
from marga import Proxy
import os

# Configure backends
proxy = Proxy(backends={
    "local": {
        "base_url": "http://127.0.0.1:8080/v1",
        "models": ["qwen2.5-coder"]
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "models": ["gpt-4o-mini", "gpt-4o"]
    }
})

# Add smart routing
@proxy.route
def smart_route(ctx):
    """Route small requests to local, large to OpenAI."""
    if ctx.estimated_tokens < 2000:
        return "local/qwen2.5-coder"
    return "openai/gpt-4o-mini"

# Add RAG hook
@proxy.before
def inject_rag_context(ctx):
    """Add relevant context before sending to LLM."""
    # In real usage, query your vector DB here
    user_msg = ctx.messages[-1]["content"]
    
    if "quantum" in user_msg.lower():
        ctx.messages.insert(0, {
            "role": "system",
            "content": "Context: Quantum computing uses qubits..."
        })

# Log responses
@proxy.after
def log_usage(ctx, response):
    """Track token usage."""
    tokens = response.get("usage", {}).get("total_tokens", 0)
    print(f"Used {tokens} tokens via {ctx.model}")
    return response

# Start server
if __name__ == "__main__":
    proxy.serve(host="127.0.0.1", port=8000)
```

Run it:
```bash
python my_proxy.py
```

## Test It

```bash
# List models
curl http://127.0.0.1:8000/v1/models

# Send request
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Check stats
curl http://127.0.0.1:8000/stats
```

## Use with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"  # marga doesn't require auth
)

response = client.chat.completions.create(
    model="gpt-4o-mini",  # Routes via your smart_route function
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

## Hook Context API

In `@proxy.before` hooks, you get `HookContext`:

```python
@proxy.before
def my_hook(ctx):
    # Access/modify messages
    ctx.messages  # list[dict]
    ctx.messages.append({"role": "system", "content": "..."})
    
    # Check request details
    ctx.model  # str: requested model
    ctx.stream  # bool: is streaming?
    ctx.estimated_tokens  # int: rough token count
    
    # Store metadata for later hooks
    ctx.metadata["my_key"] = "my_value"
```

In `@proxy.after` hooks:

```python
@proxy.after
def my_hook(ctx, response):
    # ctx: same HookContext
    # response: OpenAI-format response dict
    
    # Modify response
    response["custom_field"] = "value"
    
    # Must return response
    return response
```

## Router API

Custom routing function:

```python
@proxy.route
def my_router(ctx):
    # ctx: HookContext
    
    # Return "backend/model" string
    return "openai/gpt-4o-mini"
    
    # Or just backend name (uses first model)
    # return "openai"
    
    # Default behavior (least-used) if no router registered
```

## Endpoints

- `POST /v1/chat/completions` - Chat completions (streaming + non-streaming)
- `GET /v1/models` - List all models from all backends
- `GET /health` - Health check
- `GET /stats` - Usage statistics

## Configuration

Backend config structure:

```python
{
    "backend_name": {
        "base_url": "http://...",     # Required
        "api_key": "sk-...",           # Optional (default: "")
        "models": ["model1", "model2"] # Optional (default: [])
    }
}
```

## Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY marga/ /app/marga/
COPY my_proxy.py /app/

RUN pip install -e marga/

EXPOSE 8000
CMD ["python", "my_proxy.py"]
```

Run:

```bash
docker build -t my-marga-proxy .
docker run -p 127.0.0.1:8000:8000 my-marga-proxy
```

## Environment-Based Config

```python
import os

proxy = Proxy(backends={
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "models": os.getenv("OPENAI_MODELS", "gpt-4o,gpt-4o-mini").split(",")
    }
})
```

## Homelab Integration

### With LibreChat

In LibreChat config, add custom endpoint:

```yaml
endpoints:
  custom:
    - name: "marga"
      apiKey: "not-needed"
      baseURL: "http://127.0.0.1:8000/v1"
      models:
        default: ["gpt-4o-mini", "qwen2.5-coder"]
```

### With Continue.dev (VSCode)

In `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "marga",
      "provider": "openai",
      "model": "gpt-4o-mini",
      "apiBase": "http://127.0.0.1:8000/v1",
      "apiKey": "not-needed"
    }
  ]
}
```

### With Cursor

Settings → Models → Add Model:
- Base URL: `http://127.0.0.1:8000/v1`
- API Key: `not-needed`

## Advanced: RAG with ChromaDB

```python
import chromadb
from marga import Proxy

# Setup vector DB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("my_docs")

# Add documents
collection.add(
    documents=["Quantum computing uses qubits...", "AI is..."],
    ids=["doc1", "doc2"]
)

proxy = Proxy(backends={...})

@proxy.before
def add_rag(ctx):
    """Retrieve relevant context from ChromaDB."""
    query = ctx.messages[-1]["content"]
    results = collection.query(query_texts=[query], n_results=3)
    
    if results["documents"]:
        context = "\n".join(results["documents"][0])
        ctx.messages.insert(0, {
            "role": "system",
            "content": f"Relevant context:\n{context}"
        })
```

## Performance Tips

1. **Import time**: ~240ms (vs 380ms for full margAI)
2. **Request overhead**: <2ms per request
3. **Memory**: ~10MB baseline

For serverless/lambda:
- Import `Proxy` once globally
- Reuse same instance across requests
- Consider lazy backend connection

## Troubleshooting

**Import errors?**
```bash
uv pip install fastapi httpx uvicorn
```

**Can't reach backends?**
- Check `base_url` is correct
- Verify backend is running: `curl http://127.0.0.1:8080/v1/models`
- Check firewall rules

**No models showing?**
- Ensure `models` list is populated in backend config
- Check `/v1/models` endpoint

**Routing not working?**
- Check router function is registered with `@proxy.route`
- Verify it returns `"backend/model"` string
- Check `/stats` to see actual routing

## What's Next?

- See `example.py` for complete working example
- See `COMPARISON.md` for marga vs margAI differences
- See `test_marga.py` for more usage patterns
- Read source: only 509 lines total!

## Support

This is a minimal library by design. If you need:
- Bangtag syntax (`!app: route=...`)
- Complex config layering
- Plugin ecosystem
- Built-in diagnostics

→ Use full margAI instead.

If you need transparent routing + hooks → **marga is perfect**.
