"""CLI entry point: ``uv run margAI``."""

from __future__ import annotations

from .config import ConfigError, load_config
from .transport.fastapi import build_app


def run() -> None:
    import uvicorn

    try:
        config = load_config()
        app = build_app(config=config)
    except ConfigError as exc:
        raise SystemExit(f"margAI: config error: {exc}") from exc

    uvicorn.run(app, host=config.gateway.host, port=config.gateway.port)


if __name__ == "__main__":
    run()