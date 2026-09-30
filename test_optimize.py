#!/usr/bin/env python
"""Manual smoke test for the `optimize` marglet against a live upstream.

Not part of the pytest suite: it needs a real provider, an API key, and a
model that can actually write code. Run it by hand:

    uv run python test_optimize.py                       # uses ./margAI.toml
    uv run python test_optimize.py --config other.toml
    uv run python test_optimize.py --model margAI/openai/gpt-4o

Note the bangtag syntax: `!margAI: optimize`, not `!optimize`. The namespace
is what keeps user-typed text from accidentally firing someone's handler, so
an unnamespaced directive is (correctly) ignored -- if this script reports
"the marglet did not fire", check the namespace before anything else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from margAI import Wrapper
from margAI.config import load_config

# A deliberately slow algorithm, so there is something to optimize.
SLOW_CODE = (
    "def bubble_sort(arr):\n"
    "    for i in range(len(arr)):\n"
    "        for j in range(len(arr)-1):\n"
    "            if arr[j] > arr[j+1]:\n"
    "                arr[j], arr[j+1] = arr[j+1], arr[j]"
)

# Recorded before the directive is stripped, so a run can prove the marglet
# actually activated rather than inferring it from the answer's tone.
PROMPT = f"{SLOW_CODE}\n\nExplain what is slow about this.  !margAI: optimize"


def build(config_path: str | None) -> Wrapper:
    # `Wrapper.from_config` loads [hooks] load entries itself, and
    # margAI.toml already lists margAI.examples.optimize_hooks. Registering it
    # a second time here would be a no-op (it guards on a WeakSet) but would
    # obscure where the marglet came from.
    app = Wrapper.from_config(load_config(config_path))
    if "optimize" not in app.marglets:
        print(
            "! the optimize marglet is not registered; add "
            "`[hooks] load = [\"margAI.examples.optimize_hooks\"]` to the config",
            file=sys.stderr,
        )
    return app


async def test_non_streaming(app: Wrapper, model: str) -> int:
    print("=== non-streaming: !margAI: optimize ===")
    result = await app.complete({"model": model, "messages": [{"role": "user", "content": PROMPT}]})
    if result.status != 200:
        print(f"error: {json.dumps(result.body)}")
        return result.status
    print(result.body["choices"][0]["message"]["content"][:500])
    return result.status


async def test_streaming(app: Wrapper, model: str) -> int:
    print("\n=== streaming: !margAI: optimize ===")
    try:
        handle = await app.open_stream(
            {"model": model, "messages": [{"role": "user", "content": PROMPT}], "stream": True}
        )
    except Exception as exc:
        print(f"error opening stream: {exc}")
        return 1
    async for line in handle.lines():
        if line.strip() == "data: [DONE]":
            break
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        if payload.get("error"):
            print(f"error: {json.dumps(payload['error'])}")
            return 1
        choices = payload.get("choices") or []
        if choices:
            text = (choices[0].get("delta") or {}).get("content")
            if text:
                print(text, end="", flush=True)
    print()
    return 0


def default_model(app: Wrapper) -> str:
    """The reserved dynamic id, so the script works on any config."""
    return app.router.dynamic_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="path to a margAI.toml (default: $MARGAI_CONFIG or ./margAI.toml)")
    parser.add_argument("--model", help="a margAI/<provider>/<model> id (default: the reserved dynamic id)")
    args = parser.parse_args()

    app = build(args.config)
    model = args.model or default_model(app)
    print(f"config: {app.config.source or '<defaults>'}  model: {model}\n")

    status = asyncio.run(test_non_streaming(app, model))
    if status != 200:
        return status
    return asyncio.run(test_streaming(app, model))


if __name__ == "__main__":
    raise SystemExit(main())
