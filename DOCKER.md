# Running margAI in Docker

margAI is an OpenAI-compatible LLM gateway: it routes requests to multiple
providers, applies optimization hooks, and records telemetry.

It is **optional**. LibreChat and open-notebook talk to llama.cpp directly and
do not need it.

## Requirements

- Docker with `network_mode: host` support
- A model backend. Here that is **Windows-native llama.cpp** serving an
  OpenAI-compatible API on `127.0.0.1:8080`.

Verify the backend before starting:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

## Start

```bash
cd ~/codebox/margAI
docker compose up -d --build
curl -s http://127.0.0.1:8002/v1/models
docker compose stop          # leave runnable but idle
```

## Why host networking

margAI's upstream is Windows llama.cpp, reached at `127.0.0.1:8080`.

WSL2 **mirrored networking mode** gives WSL the same loopback as Windows, and
`network_mode: host` puts the container in that same namespace. So the
container's `127.0.0.1` *is* the address llama.cpp listens on:

```bash
docker inspect margai --format '{{.HostConfig.NetworkMode}}'   # host
```

Under the default bridge network the container gets its own, empty loopback and
`127.0.0.1:8080` would be refused. `host.docker.internal` does not help either:
under mirrored mode it resolves to the host gateway, which is not the Windows
loopback.

## Do not rebind llama.cpp to 0.0.0.0

Earlier versions of this file recommended setting the backend to `0.0.0.0` so
containers could reach it. That advice is wrong for this setup and should not be
followed.

- It is unnecessary. Host networking plus mirrored mode already reaches it.
- llama.cpp has no authentication, so a `0.0.0.0` bind publishes an open,
  unauthenticated inference endpoint to your whole LAN.

If you ever do need the backend reachable from a bridge network, the correct
fix is to keep the bind on `127.0.0.1` and run the consumer on the host network,
not to widen the backend's bind.

## Configuration

`margAI.toml` is mounted read-only. TOML is required — the loader is
`tomllib`-only and has no YAML support.

```toml
[gateway]
prefix = "margAI"             # model ids appear as margAI/<provider>/<model>
expose = "prefixed"           # prefixed | both | raw
default_provider = "llamacpp" # routes unprefixed/unknown models here
host = "127.0.0.1"            # keep off the LAN under host networking
port = 8002

[providers.llamacpp]
kind = "openai-compatible"
base_url = "http://127.0.0.1:8080/v1"
models = ["qwen2.5-coder"]
```

`host = "127.0.0.1"` matters. margAI's default is `0.0.0.0`, and under host
networking that publishes the gateway to your LAN.

### Overriding the upstream per host

Any `[providers.<name>]` block can have its URL overridden by environment
variable, so the same config works on another machine without edits:

```bash
MARGAI_PROVIDER_BASE_URL_LLAMACPP=http://other-host:8080/v1
```

Or through compose, which passes it for you:

```bash
MARGAI_LLAMACPP_URL=http://other-host:8080/v1 docker compose up -d
```

The variable name is `MARGAI_PROVIDER_BASE_URL_` plus the provider name
uppercased, with `-` replaced by `_`.

## Adding cloud providers

```toml
[providers.openai]
kind = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
models = ["gpt-4o", "gpt-4o-mini"]
default_model = "gpt-4o"
```

Pass the key through compose or an env file:

```bash
OPENAI_API_KEY=sk-... docker compose up -d
```

## Model ids

Requests reference the prefixed id, e.g. `margAI/llamacpp/qwen2.5-coder`. That
is what `curl http://127.0.0.1:8002/v1/models` returns.

To point LibreChat at margAI, its `librechat.yaml` needs:

```yaml
- name: 'margAI'
  apiKey: "not-needed"
  baseURL: "http://localhost:8002/v1"
  models:
    default: ["margAI/llamacpp/qwen2.5-coder"]
    fetch: true
```

## A completion through the gateway

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "margAI/llamacpp/qwen2.5-coder",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}]
  }'
```

## Verifying

```bash
docker ps --filter name=margai --format '{{.Names}} {{.Status}}'   # Up (healthy)
ss -tln | grep ':8002 '                                            # 127.0.0.1:8002
curl -s http://127.0.0.1:8002/v1/models                            # JSON list
```

`healthy` only proves the process is answering `/v1/models`. It does not prove
the upstream is reachable — that check is the `curl` above returning your model
id.

## Troubleshooting

**`healthy` but `/v1/models` is empty**

`models` in `margAI.toml` is a static list, so this normally means the config
failed to load. Check the log for a TOML parse error.

**Upstream connection refused**

Confirm llama.cpp is up and reachable from inside the container:

```bash
curl -s http://127.0.0.1:8080/v1/models
docker exec margai python -c \
  "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8080/v1/models').read()[:200])"
```

If the first works and the second does not, the container is not on host
networking.

**Container permanently unhealthy with `curl: not found`**

The image is `python:3.12-slim`, which has no `curl`. The healthcheck uses
`python -c` with `urllib` for exactly this reason. If you see a curl-based
healthcheck, the file was edited.

## Tests

```bash
cd ~/codebox/margAI && uv run pytest -q
```
