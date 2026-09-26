#!/usr/bin/env python
"""Test script for optimize hooks with local Ollama."""

import asyncio
from margAI import Wrapper, install_bangtags
from margAI.config import load_config


async def main():
    # Load config
    config = load_config("/home/vgana/codebox/margAI/margAI.toml")
    
    # Build wrapper
    app = Wrapper.from_config(config)
    
    # Register optimize hooks (from examples.optimize_hooks)
    from examples.optimize_hooks import register
    register(app)
    
    # Test non-streaming
    print("=== Testing non-streaming with !optimize ===")
    result = await app.complete({
        "model": "margAI/ollama/llama3.2:3b",
        "messages": [{"role": "user", "content": "def fib(n): return n if n<=1 else fib(n-1)+fib(n-2) !optimize"}]
    })
    
    if result.status == 200:
        content = result.body["choices"][0]["message"]["content"]
        print(content[:500])
    else:
        print(f"Error: {result.body}")
    
    # Test streaming
    print("\n=== Testing streaming with !optimize ===")
    handle = await app.open_stream({
        "model": "margAI/ollama/llama3.2:3b",
        "messages": [{"role": "user", "content": "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]  !optimize"}],
        "stream": True
    })
    
    async for line in handle.lines():
        if line.strip() == "data: [DONE]":
            break
        if line.startswith("data: "):
            import json
            payload = json.loads(line[6:])
            choices = payload.get("choices") or []
            if choices:
                content = choices[0].get("delta", {}).get("content")
                if content:
                    print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())