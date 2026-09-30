"""Minimal Gateway Example - The simplest possible margAI setup.

This shows the absolute minimum to get started:
1. Create a Gateway
2. Define tags with @app.tag()
3. Serve

No configuration complexity, no hidden magic, just decorators.
"""

from margAI import Gateway

# Create gateway - reads ./margAI.toml automatically
app = Gateway()

# Define tags as simple decorated functions
@app.tag("summarize")
def summarize(ctx, tag):
    """Add instruction to summarize."""
    ctx.add_system("Provide a concise summary in 2-3 sentences.")

@app.tag("expand")
def expand(ctx, tag):
    """Add instruction to expand."""
    ctx.add_system("Provide a detailed, comprehensive response with examples.")

@app.tag("formal")
def formal(ctx, tag):
    """Make response more formal."""
    ctx.add_system("Respond in a formal, professional tone.")

@app.tag("casual")
def casual(ctx, tag):
    """Make response more casual."""
    ctx.add_system("Respond in a casual, friendly tone.")

# Optional: hooks that run on every request
@app.on_request
def log_request(ctx):
    """Log every incoming request."""
    model = ctx.body.get("model", "unknown")
    msg = ctx.user_message()[:60]
    print(f"→ {model}: {msg}...")

@app.on_response
def log_response(payload, ctx):
    """Log every response."""
    tokens = payload.get("usage", {}).get("total_tokens", 0)
    print(f"← {tokens} tokens\n")
    return payload

if __name__ == "__main__":
    print(f"""
Minimal Gateway Example
=======================

Tags defined:
  - summarize: Request brief response
  - expand: Request detailed response
  - formal: Use formal tone
  - casual: Use casual tone

Example prompts (use in your LLM client):

  !{app.namespace}: summarize
  Explain quantum computing

  !{app.namespace}: expand, formal
  What is Python?

Server starting on http://{app.wrapper.config.gateway.host}:{app.wrapper.config.gateway.port}
""")
    
    app.serve()
