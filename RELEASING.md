# Releasing margAI

Written for the first release, and kept for every release after it. Follow it in
order. The irreversible steps are marked.

## The two facts that matter

1. **A published version cannot be changed or withdrawn.** You can yank it, but
   the files stay downloadable forever and the version number is burned. If
   `0.1.0` has a bug, the fix is `0.1.1`.
2. **A PyPI project name is permanent.** The first upload claims it. `margAI`
   and `margai` are the same name to PyPI — it normalises to lowercase — so
   `margai` is what gets claimed either way.

## One-time setup

- [ ] **PyPI account.** <https://pypi.org/account/register/>
- [ ] **2FA enabled.** Required by PyPI before *any* upload or management
      action. Use a security key or TOTP app, not SMS.
- [ ] **TestPyPI account** (separate). <https://test.pypi.org/account/register/>
      This is a distinct account with a distinct 2FA.
- [ ] **Repository pushed to GitHub** and public. Trusted publishing cannot be
      configured against a local-only repo.
- [ ] **Check the name is still free:** `curl -o /dev/null -w "%{http_code}" \
      https://pypi.org/pypi/margai/json` → `404` means unclaimed. It was
      unclaimed when this was written; re-check immediately before you publish.
- [ ] **Trusted publisher registered on PyPI** — account → *Publishing* → add a
      GitHub publisher:

      | Field | Value |
      | --- | --- |
      | Owner | `evolvedHow` |
      | Repository name | `margAI` |
      | Workflow filename | `release.yml` |
      | Environment name | `pypi` |

      Take owner and repo from the repo's settings page, not from memory.
- [ ] **GitHub environment `pypi` created** (Settings → Environments). It can
      have no reviewers; it exists so the upload token is scoped to one job.

## Before every release

- [ ] `uv run pytest -q` — all green, **zero warnings**
- [ ] `uv run ruff check src tests`
- [ ] `uv run mypy src/margAI`
- [ ] Version bumped in `src/margAI/__init__.py` (the only place it lives —
      `pyproject.toml` reads it via `[tool.hatch.version]`, so the filename, the
      metadata, and `margAI.__version__` cannot disagree)
- [ ] `CHANGELOG.md` has a dated section for this version, and the *Unreleased*
      heading is reset.
- [ ] `git tag -a v<version> -m "v<version>"` and push the tag

## Dry run on TestPyPI first

Do this at least once. It is the only way to find out that your publishing setup
is wrong before the real name is claimed.

```bash
uv build
uvx twine upload --repository testpypi dist/*
```

Then install it *into a clean environment* and actually use it:

```bash
uv venv /tmp/check && VIRTUAL_ENV=/tmp/check uv pip install \
  --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple \
  margai
/tmp/check/bin/python -c "import margAI; print(margAI.__version__)"
/tmp/check/bin/margAI --help
```

For the CI path, add a temporary tag like `v0.1.0-rc1` and point
`gh-action-pypi-publish` at TestPyPI with
`repository-url: https://test.pypi.org/legacy/`. Delete the tag afterwards.

## Publishing for real

- [ ] Confirm the tag is on the commit you tested.
- [ ] `git push origin main && git push origin v<version>`
- [ ] Watch the run: <https://github.com/evolvedHow/margAI/actions>
      `ci.yml` must pass before `release.yml` will publish.
- [ ] If publishing fails with 401, the trusted-publisher details do not match —
      almost always the workflow filename or the environment name.

## After it lands

- [ ] `pip install margai` in a clean venv and run it. Not the same as
      TestPyPI; this is the one people will actually use.
- [ ] <https://pypi.org/project/margai/> renders correctly — description,
      author, license, README, and no broken relative links.
- [ ] The GitHub tag has release notes, or a link to the changelog section.
- [ ] Announce it if you want to.

## If something goes wrong

| Symptom | Do this |
| --- | --- |
| 401 from the publish step | Trusted publisher mismatch. Owner, repo, `release.yml`, `pypi`. |
| `409 Conflict` | That version already exists. Versions are immutable — bump the number. |
| Name taken | You cannot publish under it. If someone else holds it, you would have to use a different name; there is no transfer without their cooperation. |
| `twine check` warnings | Fix before uploading. `--strict` catches what plain `twine check` misses. |
| Bad release published | Yank it (`pypi yank`), publish a fixed version. Do not reuse the number. |
| Secret leaked in a log | Revoke it first, then investigate. |

## Routine releases

The checkboxes above are the whole process. Nothing needs re-deciding except:

- Never reuse a version number.
- Keep `CHANGELOG.md` current as you go, not at tag time.
- Bump the patch number for fixes, minor for features, and expect to break
  things freely until 1.0.
