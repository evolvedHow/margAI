"""RAG Gateway Example - Shows vector DB, caching, and smart routing.

This demonstrates:
- Vector DB lookup via @app.tag("rag")
- Response caching via @app.tag("cache")
- Smart routing based on query complexity
- All using simple decorators
"""

from pathlib import Path
import hashlib
import json

from margAI import Gateway


# Mock implementations (replace with real ones)
class MockVectorDB:
    """Mock vector database for demonstration."""
    
    def __init__(self):
        self.docs = {
            "quantum": [
                "Quantum entanglement is a phenomenon where particles become correlated.",
                "Quantum superposition allows particles to exist in multiple states.",
                "Quantum computing uses qubits instead of classical bits.",
            ],
            "python": [
                "Python is a high-level programming language.",
                "Python uses indentation for code blocks.",
                "Python has a large ecosystem of libraries.",
            ],
        }
    
    def search(self, query, top_k=3):
        """Simulate vector search."""
        query_lower = query.lower()
        for topic, docs in self.docs.items():
            if topic in query_lower:
                return docs[:top_k]
        return ["No relevant documents found."]


class SimpleCache:
    """Simple in-memory cache."""
    
    def __init__(self):
        self._cache = {}
    
    def key(self, messages):
        """Generate cache key from messages."""
        content = json.dumps([m for m in messages if m.get("role") != "system"])
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key):
        return self._cache.get(key)
    
    def set(self, key, value):
        self._cache[key] = value


# Initialize resources
vector_db = MockVectorDB()
cache = SimpleCache()


# Create the gateway. Resolved next to this file so the example runs from any
# working directory, not only the repo root.
app = Gateway(str(Path(__file__).parent / "simple_gateway.toml"))


# Define tags
@app.tag("rag")
def add_rag_context(ctx, tag):
    """Add vector DB context to the prompt.
    
    Usage: !app: rag=quantum-computing
    """
    query = tag.value or ctx.last_user_message() or ""
    if not query:
        return
    
    # Search vector DB
    docs = vector_db.search(query, top_k=3)
    context = "\n\n".join(f"- {doc}" for doc in docs)
    
    # Add to system prompt
    ctx.add_system_prompt(f"Relevant context:\n{context}")
    
    print(f"🔍 RAG: Added {len(docs)} documents for query: {query}")


@app.tag("cache")
def check_cache(ctx, tag):
    """Check if we have a cached response.
    
    Usage: !app: cache
    """
    messages = ctx.body.get("messages", [])
    key = cache.key(messages)
    
    if cached := cache.get(key):
        print(f"💾 Cache HIT: {key[:8]}")
        # TODO: Need to implement short-circuit
        ctx.state["cached"] = cached
    else:
        print(f"💾 Cache MISS: {key[:8]}")
        ctx.state["cache_key"] = key


@app.tag("smart")
def smart_routing(ctx, tag):
    """Route based on query complexity.
    
    Usage: !app: smart
    """
    message = ctx.last_user_message() or ""
    word_count = len(message.split())
    
    # Simple heuristic: short queries go to fast model
    if word_count < 20:
        model = "app/local/llama3.2-1b"
        print(f"🚀 Smart route: FAST model ({word_count} words)")
    else:
        # For this example, we only have local models
        model = "app/local/gemma-4-E2B"
        print(f"🧠 Smart route: CAPABLE model ({word_count} words)")
    
    ctx.body["model"] = model


@app.tag("fast")
def route_fast(ctx, tag):
    """Route to fastest model.
    
    Usage: !app: fast
    """
    ctx.body["model"] = "app/local/llama3.2-1b"
    print("🚀 Route: fast (llama3.2-1b)")


@app.tag("summarize")
def summarize(ctx, tag):
    """Request a summary response.
    
    Usage: !app: summarize
    """
    ctx.add_system_prompt("Provide a concise summary in 2-3 sentences.")
    print("📝 Mode: summarize")


@app.tag("expand")
def expand(ctx, tag):
    """Request a detailed response.
    
    Usage: !app: expand
    """
    ctx.add_system_prompt("Provide a detailed, comprehensive response with examples.")
    print("📚 Mode: expand")


# Global hooks
@app.on_request
def log_request(ctx):
    """Log every request."""
    model = ctx.body.get("model", "unknown")
    messages = ctx.body.get("messages", [])
    last_msg = messages[-1].get("content", "")[:60] if messages else ""
    
    tags = [t.name for t in ctx.tags] if hasattr(ctx, 'tags') else []
    tags_str = f" [{', '.join(tags)}]" if tags else ""
    
    print(f"\n→ Request{tags_str}: {model}")
    print(f"  Message: {last_msg}...")


@app.on_response
def log_and_cache_response(payload, ctx):
    """Log response and update cache."""
    usage = payload.get("usage", {})
    tokens = usage.get("total_tokens", 0)
    
    # Cache the response if cache tag was used
    if "cache_key" in ctx.state:
        key = ctx.state["cache_key"]
        cache.set(key, payload)
        print(f"💾 Cached response: {key[:8]}")
    
    print(f"← Response: {tokens} tokens\n")
    return payload


if __name__ == "__main__":
    print("""
RAG Gateway Example
===================

Available tags:
  rag=<query>  - Add vector DB context
  cache        - Use response cache
  smart        - Smart routing (based on complexity)
  fast         - Route to fastest model
  summarize    - Request brief response
  expand       - Request detailed response

Example prompts:

  1. RAG with caching:
     !app: rag=quantum, cache, smart
     What is quantum entanglement?

  2. Fast summarization:
     !app: fast, summarize
     Explain Python

  3. Detailed with context:
     !app: rag, expand
     Tell me about quantum computing

Server starting...
""")
    
    app.serve()
