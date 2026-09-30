"""Simple Gateway API Example - CrewAI-style decorator approach.

This demonstrates the simplified Gateway API:
- One class: Gateway
- One decorator: @app.tag()
- No hidden magic
- Clear routing semantics
"""

from margAI import Gateway


# Example usage
if __name__ == "__main__":
    # Create gateway - reads ./simple_gateway.toml
    app = Gateway("examples/simple_gateway.toml")
    
    # Define tags with decorators
    @app.tag("summarize")
    def summarize(ctx, tag):
        """Add a system prompt to summarize."""
        ctx.add_system_prompt("Provide a concise summary of your response.")
    
    @app.tag("expand")
    def expand(ctx, tag):
        """Add a system prompt to expand."""
        ctx.add_system_prompt("Provide a detailed, comprehensive response.")
    
    @app.tag("translate")
    def translate(ctx, tag):
        """Translate to the specified language."""
        language = tag.value or "French"
        ctx.add_system_prompt(f"Translate your response to {language}.")
    
    @app.tag("fast")
    def route_fast(ctx, tag):
        """Route to fast local model."""
        # This is a simplified example - actual routing is more complex
        ctx.body["model"] = "local/llama3.2-1b"
    
    # Hooks that run on every request
    @app.on_request
    def log_request(ctx):
        model = ctx.body.get("model", "unknown")
        messages = ctx.body.get("messages", [])
        last_msg = messages[-1].get("content", "") if messages else ""
        print(f"→ {model}: {last_msg[:60]}...")
    
    @app.on_response
    def log_response(payload, ctx):
        usage = payload.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        print(f"← {tokens} tokens")
        return payload
    
    print(f"""
Simple Gateway Example
======================
Tags defined: summarize, expand, translate, fast

Example prompts:
  !{app.namespace}: summarize
  Explain quantum computing

  !{app.namespace}: translate=Spanish
  Hello, how are you?

  !{app.namespace}: fast, summarize
  Quick answer: what is Python?

Server starting on http://{app.wrapper.config.gateway.host}:{app.wrapper.config.gateway.port}
""")
    
    app.serve()
