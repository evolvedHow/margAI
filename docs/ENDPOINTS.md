# The routed surface

margAI routes five endpoint kinds. Every one of them is backed by a real
`Provider` method and covered by the test suite.

| kind | route | stream | wire format |
| --- | --- | --- | --- |
| `chat` | `POST /v1/chat/completions` | yes | JSON |
| `completions` | `POST /v1/completions` | yes | JSON |
| `embeddings` | `POST /v1/embeddings` | no | JSON |
| `images_generations` | `POST /v1/images/generations` | no | JSON |
| `audio_transcriptions` | `POST /v1/audio/transcriptions` | no | multipart |
| — | `GET /v1/models`, `GET /v1/models/{id}` | — | served by the wrapper |
| — | `GET /v1/margAI/tags` | — | served by the wrapper |

`GET /v1/models` always lists the reserved `margAI/<dynamic_model>` id, so a
client using a model dropdown (LibreChat) can reach the dynamic router at all.

`GET /v1/margAI/tags` is introspection rather than routing: it reports the
registered tags by namespace, the configured virtual models, providers, packs,
and config provenance — the same payload as `wrapper.describe()`. It exists
because the alternative was answering "what bangtags does this gateway
actually understand?" by reading its source.

`margAI doctor` is the same idea for a terminal, with a pass/fail exit code
for CI. It is a CLI, not a route, so it can be run against a config that would
not start a server.

## Why the surface is small

This list used to be 41 kinds — files, batches, assistants, threads,
vector stores, fine-tuning, image edits, audio speech, moderations. All of
it was copy-paste: 90 `raise ApiError(404)` stubs on the `Provider` ABC, 45
near-identical passthroughs on `OpenAICompatProvider`, and two 41-entry
dicts in `wrapper.py` that had to be kept in sync with both *and* with the
FastAPI routes. It was also broken in ways nobody could see, because no test
exercised any of it:

- every model-less endpoint (`files`, `threads`, `assistants`, …) died at
  `router.resolve()` with `400 No model specified`, since routing always
  demands a model;
- four GET routes were mapped to POST upstream methods, so
  `GET /v1/fine-tuning/jobs/{id}` *created* a fine-tuning job;
- query parameters (`limit`, `after`) were collected into a body that GET
  requests then dropped on the floor — pagination never worked;
- uploads were impossible: `PreparedRequest` had no way to carry multipart,
  so every file/audio/image-edit route rejected real clients;
- binary responses (audio speech, file downloads) were JSON-encoded text.

All of that was ~1,750 lines. It is gone. A proxy surface you cannot
test is a liability, not a feature.

## Adding a kind

Four edits, in this order. Keep them together or the surface rots again.

1. **`providers/openai_compat.py`** — on `OpenAICompatProvider`, add
   `prepare_<kind>` and `parse_<kind>_response`. For a JSON passthrough:

   ```python
   def prepare_moderations(self, ctx: Any) -> PreparedRequest:
       return self._json(ctx.body, "moderations")

   def parse_moderations_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
       return _openai_body(resp, {"results": []})
   ```

   Add `parse_<kind>_chunk` only if the endpoint actually streams, and
   register the kind in `Wrapper._STREAMABLE_KINDS` and `_KIND_PARSE_CHUNK`.

   For multipart, use `self._form_value(ctx, key, default)` and pass
   `files=` on `PreparedRequest` — never set `Content-Type` yourself, the
   transport has to add the boundary. A request may carry a JSON body *or*
   `data`/`files`, never both; the transport rejects that combination.

   For a wire format that is neither JSON nor multipart, or an endpoint that
   does not route by model (`/v1/files`), write the method by hand — but then
   also make the *route* skip model resolution. That is the trap the old
   surface fell into: see "Model-less kinds" below.

   Add the 404-raising pair to the `Provider` base class as well, so any
   provider that doesn't implement the kind degrades to a 404. See "Providers
   that don't speak a kind" below.

2. **`wrapper.py`** — add the two entries to `_KIND_PREPARE` / `_KIND_PARSE`
   next to the existing ones. These dicts are the routing table's source of
   truth; a kind in one and not the other is a bug.

3. **`transport/fastapi.py`** — add the pydantic request model and the route,
   forwarding through `_forward(..., kind="<kind>")`.

4. **`docs/ENDPOINTS.md`** — add the row to the table above.

Then add a test. `tests/test_fastapi.py` drives the adapter through
`TestClient` with the `FakeTransport` from `tests/conftest.py`; assert on the
upstream `PreparedRequest` that came out the other side, not just the status
code. That assertion is what would have caught all four bugs above.

The declarative endpoint table mentioned in the roadmap would replace steps
1–3 with one declaration per kind. Until it exists, these four dicts and two
provider methods are the contract, and the test is what keeps them honest.

## Model-less kinds

A kind whose request has no `model` (`/v1/files`, `/v1/batches`) cannot be
routed by the current `ModelRouter`, which requires one. If you add such a
kind you must also pick how it picks a provider:

- require the client to send an explicit `margAI/<provider>/...` id anyway
  (current behaviour, but the client will not do it);
- give the kind its own provider-selection hook;
- or serve it from a single designated provider, bypassing the router.

The old code did none of these and shipped 400s instead.

## Providers that don't speak a kind

`Provider` keeps `raise ApiError(404, "Provider 'x' does not support ...")`
fallbacks for every non-chat kind, so a provider like Anthropic degrades
cleanly. Both halves of each kind's pair need one — `prepare_<kind>` *and*
`parse_<kind>_response` — or the failure lands a step later than the routing
decision and the client sees a 500 for what is really a 404.

`Wrapper._provider_method` is the backstop for anything that slips through: a
kind in `_KIND_PREPARE` that a provider has not implemented degrades to a 404
naming the provider, not an `AttributeError` that surfaces as a redacted 500.
New kinds should still get the explicit fallback.
