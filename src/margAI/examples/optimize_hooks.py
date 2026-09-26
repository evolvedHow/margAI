"""Example: Intercept calls with !optimize bangtag, add context, and strip on response."""

from __future__ import annotations

from typing import Any

from margAI import Wrapper, install_bangtags


OPTIMIZE_SYSTEM_PROMPT = """You are an expert code optimizer. When the user adds !optimize to their prompt,
they want you to focus on:
- Performance improvements (time/space complexity)
- Memory efficiency
- Clean, maintainable code
- Modern best practices

Provide the optimized code with brief explanations of what changed and why."""


def register(app: Wrapper) -> Wrapper:
    """Register the optimize hooks on the wrapper."""
    if getattr(app, "_optimize_registered", False):
        return app

    # 1. Install bangtag parsing (looks for !margAI: optimize in user messages)
    install_bangtags(app)

    # 2. BEFORE hook: runs when !optimize tag is present
    # Adds the optimization system prompt and strips the tag from user message
    @app.before("optimize")
    def optimize_before(ctx: Any, tag: Any) -> None:
        # Add the optimization system prompt (prepended to existing system prompt if any)
        ctx.add_system_prompt(OPTIMIZE_SYSTEM_PROMPT, prepend=True)

        # Store original user message for reference
        original = ctx.last_user_message()
        if original:
            ctx.state["optimize_original"] = original

    # 3. AFTER hook: runs on non-streaming response
    # Strips any internal markers and returns clean response
    @app.after("optimize")
    def optimize_after(payload: dict, ctx: Any, tag: Any) -> dict:
        content = payload.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            # Remove any internal markers we might have added
            payload["choices"][0]["message"]["content"] = content.strip()
        return payload

    # 4. STREAM hook: runs per SSE chunk on streaming responses
    # Marks first chunk and strips on final chunk
    @app.stream("optimize")
    def optimize_stream(chunk: dict, ctx: Any, tag: Any, scratch: dict) -> dict | None:
        # Mark first chunk so we know it's the start
        if not scratch.get("first_chunk_seen"):
            scratch["first_chunk_seen"] = True
            # Could add a prefix here if needed

        # On final chunk, could do cleanup
        if chunk.get("choices", [{}])[0].get("finish_reason") == "stop":
            scratch["stream_complete"] = True

        return chunk

    # 5. ERROR hook: handles failures gracefully for optimize calls
    @app.error("optimize")
    def optimize_error(exc: Exception, ctx: Any, tag: Any) -> dict | None:
        # Return a clean error message instead of raw exception
        return {
            "error": {
                "message": f"Optimization request failed: {str(exc)}",
                "type": "optimization_error",
                "param": None,
                "code": None
            }
        }

    app._optimize_registered = True
    return app


# --- Standalone usage example ---
if __name__ == "__main__":
    import asyncio
    from margAI.config import load_config

    # Load config from margAI.toml
    config = load_config()

    # Build wrapper and register hooks
    app = Wrapper.from_config(config)
    register(app)

    # Example: Call with !optimize bangtag
    async def main():
        # Non-streaming call
        result = await app.complete({
            "model": "margAI/openai/gpt-4o",  # uses configured provider
            "messages": [
                {"role": "user", "content": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)  !optimize"}
            ]
        })
        print("=== Non-streaming response ===")
        print(result.body["choices"][0]["message"]["content"])

        print("\n=== Streaming response ===")
        # Streaming call
        handle = await app.open_stream({
            "model": "margAI/openai/gpt-4o",
            "messages": [
                {"role": "user", "content": "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]  !optimize"}
            ],
            "stream": True
        })
        async for line in handle.lines():
            print(line, end="")

    asyncio.run(main())