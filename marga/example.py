"""Example usage of marga."""

import os
from marga import Proxy

# Create proxy with multiple backends
proxy = Proxy(backends={
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "models": ["gpt-4o", "gpt-4o-mini"]
    },
    "local": {
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "",  # Local usually doesn't need key
        "models": ["qwen2.5-coder", "llama3.2-1b"]
    }
})


# Example 1: Custom routing logic
@proxy.route
def smart_route(ctx):
    """Route based on estimated complexity.
    
    - Small requests (<1000 tokens) → local model
    - Large requests → OpenAI mini
    """
    if ctx.estimated_tokens < 1000:
        return "local/qwen2.5-coder"
    return "openai/gpt-4o-mini"


# Example 2: RAG-style prompt injection
@proxy.before
def add_rag_context(ctx):
    """Inject context from vector DB before LLM.
    
    In real usage, you'd query your vector DB here.
    """
    # Example: Add system message with retrieved context
    if any("quantum" in msg.get("content", "").lower() for msg in ctx.messages):
        ctx.messages.insert(0, {
            "role": "system",
            "content": (
                "Relevant context from knowledge base:\n"
                "- Quantum entanglement is a quantum mechanical phenomenon...\n"
                "- Use this context to answer the user's question."
            )
        })
    
    # Store metadata for later use
    ctx.metadata["rag_injected"] = True


# Example 3: Response processing
@proxy.after
def log_and_enhance(ctx, response):
    """Log usage and optionally enhance response."""
    usage = response.get("usage", {})
    print(f"[STATS] Tokens: {usage.get('total_tokens', 0)} | Model: {ctx.model}")
    
    # You could modify response here
    # response["choices"][0]["message"]["content"] += "\n\n[Powered by margAI-lite]"
    
    return response


# Example 4: Conditional processing
@proxy.before
def enforce_system_prompt(ctx):
    """Ensure every request has a system prompt."""
    has_system = any(msg.get("role") == "system" for msg in ctx.messages)
    
    if not has_system:
        ctx.messages.insert(0, {
            "role": "system",
            "content": "You are a helpful, concise assistant."
        })


if __name__ == "__main__":
    # Start server on localhost:8000
    print("Starting marga proxy on http://127.0.0.1:8000")
    print("\nEndpoints:")
    print("  POST /v1/chat/completions  - Chat completions (OpenAI compatible)")
    print("  GET  /v1/models            - List available models")
    print("  GET  /health               - Health check")
    print("  GET  /stats                - Usage statistics")
    print("\nPress Ctrl+C to stop")
    
    proxy.serve(host="127.0.0.1", port=8000)
