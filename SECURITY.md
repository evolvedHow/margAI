# Security Policy

## Reporting a vulnerability

**Email vish.ganapathy@gmail.com. Please do not open a public issue for a
security problem.**

A public issue means everyone knows about the bug before there is a fix, and it
is very hard to walk back. If you have already opened one, say so in the email —
that is not held against you, it just changes what we can do.

What helps a report be useful:

- What an attacker can do, concretely. "Can read other users' API keys" is
  actionable; "seems insecure" is not.
- The margAI version (`pip show margai`), Python version, and whether you
  installed from PyPI or a git checkout.
- Steps to reproduce, or a proof of concept.
- Whether it is already public anywhere.

You will get an acknowledgement within a few days. Fixes for anything with a
plausible exploit path land quickly; slower reports are triaged, not ignored.

## Scope

**In scope** — anything in `margAI`:

- The gateway and its request handling, including header and path parsing.
- Authentication, if you have put one in front of it. margAI deliberately ships
  no auth of its own, so this is usually about how the transport passes
  credentials through, or a header it forwards upstream that it should not.
- Anything that reads a file, resolves a path, or executes a configured value.
  `hooks.load` in the config imports a Python module by name, which is
  powerful on purpose: **treat your `margAI.toml` as executable code.** Do not
  load a config from a source you do not control.
- Packs. A pack runs in-process with the same privileges as the gateway, so
  install packs you would be willing to `pip install` yourself.

**Out of scope:**

- Running margAI in front of a model you do not control, and the model
  misbehaving.
- Prompt injection in user prompts. The bangtag layer parses a directive out of
  the last user message; that is the design, and the escaping is documented. If
  you find a way to make it do something outside the documented grammar, that
  *is* in scope.
- Denial of service from your own configuration — an unreachable upstream, a
  `timeout` set too low.
- Reports from automated scanners with no demonstrated impact.
- Missing hardening headers when you are the one deploying the HTTP server.
  margAI is a library and an ASGI app; TLS, HSTS and rate limiting belong at
  your reverse proxy.

## Supported versions

margAI is pre-1.0. Fixes land on `main` and in the next release; there are no
long-term-support branches yet.

| Version | Supported |
| --- | --- |
| 0.1.x | yes |
| < 0.1 | no |

If you cannot upgrade, say so in the report and I will tell you honestly
whether a backport makes sense.

## Notes for operators

- margAI forwards `Authorization` to whatever upstream you configure. It does
  not log it, but your reverse proxy and access logs might.
- `/v1/models` and `/v1/margAI/tags` are unauthenticated introspection. If
  provider names and tag names are sensitive, restrict them at the proxy.
- Telemetry with `emit = "log"` records model, token counts, and cost to your
  logs. `emit = "file"` writes the same to disk. Neither records prompt or
  completion content — that is worth confirming against your own compliance
  requirements rather than taking this file's word for it.
