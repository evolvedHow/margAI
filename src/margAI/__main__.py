"""CLI entry point: ``margAI`` serves, ``margAI doctor`` diagnoses."""

from __future__ import annotations

import sys
from collections.abc import Sequence

USAGE = """usage: margAI [serve]     start the gateway (default)
       margAI doctor      diagnose the wiring and exit

Run `margAI doctor --help` for its options.
"""


def run(argv: Sequence[str] | None = None) -> int:
    """Dispatch the CLI. A subcommand rather than a flag, so `margAI doctor`
    can be run from a container entrypoint without a config file to hand."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "doctor":
        from .doctor import main as doctor_main

        return doctor_main(args[1:])
    if args and args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if args and args[0] not in ("serve",):
        print(USAGE, file=sys.stderr)
        return 2
    return _serve()


def _serve() -> int:
    import uvicorn

    from .config import ConfigError, load_config
    from .transport.fastapi import build_app

    try:
        config = load_config()
        app = build_app(config=config)
    except ConfigError as exc:
        raise SystemExit(f"margAI: config error: {exc}") from exc

    uvicorn.run(app, host=config.gateway.host, port=config.gateway.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
