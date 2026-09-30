# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-09-30

First release.

### Added

- **OpenAI-compatible API.** `POST /v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/v1/images/generations`, and `/v1/audio/transcriptions`
  (multipart), plus `GET /v1/models` and `GET /v1/models/{model_id}`. Any OpenAI
  SDK can point at the gateway unchanged.
- **Generated API documentation.** Every route, request field, response shape
  and error status is described in the OpenAPI document, so `/docs` (Swagger UI)
  and `/redoc` are the reference rather than a summary of it.
- **Bangtags.** A leading `!<namespace>: <tag>` directive in the last user message
  activates a tag handler and is stripped before the prompt reaches the provider.
  `GET /v1/<namespace>/tags` reports what is actually registered, read from the
  live registry rather than from the config.
- **Built-in tag pack.** `route=`, `provider=`, `model=`, `not=`, `only=`,
  `cost=` and `think=`, installed through the same pack mechanism third-party
  packs use, and opt-out-able by name.
- **Dynamic routing and virtual models.** `<namespace>/dynamic` hands each call
  to a selector chain; `[models.*]` declares a virtual model whose policy picks
  a target per call by strategy (`first`, `cheapest`, `round_robin`,
  `least_used`, `highest_balance`).
- **Hook pipeline.** `before`, `after`, `stream` and `error` phases, plus
  handler groups. A `before` handler can take a call over entirely with
  `ctx.route_to()`, `ctx.respond()` or `ctx.error()`.
- **Packs.** Third-party tag packs are discovered through the `margAI.packs`
  entry point group; `margAI doctor` reports what installed and what failed.
- **Telemetry.** Optional token, cost, latency and route-reason accounting with
  `log`, `callback` and `file` sinks.
- **`margAI` CLI.** `margAI serve` and `margAI doctor`, the latter with a
  pass/fail exit code for CI.

[Unreleased]: https://github.com/evolvedHow/margAI/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/evolvedHow/margAI/releases/tag/v0.1.0
