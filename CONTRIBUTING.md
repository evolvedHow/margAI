# Contributing to margAI

Thanks for considering it. margAI is small and young, so the bar for a change
getting in is mostly: does it work, and can you show that it does.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## The short version

```bash
uv sync --all-extras --dev    # set up
uv run pytest -q              # 528 tests, ~2s
uv run ruff check src tests   # lint
uv run mypy src/margAI        # types
```

All three run in CI on every push, on Python 3.12 and 3.13. Nothing gets merged
with a red one.

## Getting set up

You need [uv](https://docs.astral.sh/uv/). Python 3.12 is the floor
(`requires-python`); 3.13 is tested too.

```bash
git clone https://github.com/evolvedHow/margAI.git
cd margAI
uv sync --all-extras --dev
uv run pytest -q
```

## Layout

Three layers, each depending only on the one above it:

| Layer | Where | Rule |
| --- | --- | --- |
| Core | `src/margAI/core/` | pure stdlib, no I/O, no third-party imports. Must stay portable to a Worker. |
| Orchestration | `src/margAI/wrapper.py`, `routing.py`, `bangtag.py` | dispatches and parses. Must not *know* any tag by name. |
| Edges | `src/margAI/providers/`, `transport/`, `packs.py` | everything that touches the network or a config file. |

`tests/test_tags.py` enforces the middle row: a list of orchestration modules
is asserted to contain no tag names. If you find yourself wanting
`if name == "route"` in `wrapper.py`, the answer is a registry lookup, not a
branch. That test exists so the answer stays the same way in six months.

## Changing something

1. **Open an issue first** for anything that changes a public API, the config
   format, or the route surface. It is much cheaper to agree on a shape than to
   rebase a finished implementation.
2. **Branch** off `main` with a descriptive name.
3. **Write a test that fails** before you write the fix. If you can't, say so
   in the PR — "I could not find a way to demonstrate this" is a legitimate
   thing to write.
4. **Keep the suite green.** `pytest`, `ruff`, `mypy` — all three.
5. **Open a PR** describing what changed and why.

### Style

- `ruff check` passes. That is the whole of it.
- **Do not run `ruff format`.** The project lints with ruff but has never
  adopted its formatter, so a formatting pass would rewrite most of the tree and
  bury the real change. Match the surrounding code by hand.
- Type annotations on new public functions. `mypy` is clean and should stay
  that way.
- Docstrings say *why*, not *what*. If a comment restates the line below it,
  delete it.

### Tests

- Tests live in `tests/`, mirroring the module they cover.
- `conftest.py` holds the fakes: `FakeTransport`, `build_wrapper`, `make_wrapper`.
  Use them instead of reaching for `unittest.mock` — a scripted transport keeps
  the test about the pipeline rather than about patching.
- A test that documents a bug should say what the bug was, in the docstring.
  Six months from now that sentence is the most valuable line in the file.
- `tests/test_readme.py` runs the examples. If you change the README, the
  examples directory, or a decorator name, expect it to fail and tell you which
  one you broke.

## Adding a tag

Tags go through the registry — never as a special case. Register a handler:

```python
@app.tag("summarize")
def summarize(ctx, tag):
    ctx.add_system("Provide a brief summary.")
```

That is the whole mechanism. If you find a place that needs to know `summarize`
exists by name, that is a design smell, and the "no hardcoded tags" test will
say so.

For a group of hooks that travel together, use a
[marglet](README.md#marglets) instead of several tags.

## Adding a pack

Third-party tag packs are discovered through the `margAI.packs` entry point, so
a pack is just a module with an `install` function and a declared entry point:

```toml
[project.entry-points."margAI.packs"]
my_pack = "mypack:install"
```

A broken pack must not take the gateway down with it — it gets recorded as a
failure and `margAI doctor` reports it. Please keep it that way; a pack that
raises at import should still leave the server serving.

## Commits and pull requests

- Write commit messages in the imperative: "Add cost ceiling to selectors".
- One logical change per PR. Mechanical reformatting goes in its own commit so it
  can be skipped when reviewing.
- CI must be green. If it is red, say why rather than asking for a merge.

## Licensing

margAI is Apache-2.0 (see `LICENSE`). By contributing you agree that your
contribution is licensed under the same terms, and that you have the right to
license it — section 5 of the license says so, but it is worth stating here
because it is the one thing a contributor cannot negotiate later. If you are
contributing work you did not author, say so in the PR.

Contributors are also covered by the patent grant in section 3, so your
contributions come with the same protection the project gets.

There is no CLA and no DCO — opening a PR is the license.

## Code of Conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Please do not open a public issue for a security problem. Email
**vish.ganapathy@gmail.com** instead, and please do not disclose it publicly
until there is a fix. See [SECURITY.md](SECURITY.md) for what is in scope and
what to include.

## Releasing

Maintainer-only, but the process is written down in
[RELEASING.md](RELEASING.md) — including the one-time PyPI setup and what to do
when an upload fails.
