"""Bangtag directives -- an optional, separable layer of margAI.

A *bangtag* is a directive embedded in a user's message that changes how that
one call is handled, without touching configuration::

    !margAI: refine, route=llama3.2:3b
    What is the capital of France?

    !margAI: think !sc: audit region=eu
    Review the attached supplier contract.

**Directives go at the front.** The leading block of directive lines is the
directive region; everything after it is the prompt, preserved verbatim. That
is not a style preference, it is what makes the grammar unambiguous: a
directive body is a whitespace-separated token list, so with the directive at
the *end* of a message it cannot tell where the tags stop and the prose
starts -- ``!margAI: think please review this`` parses as seven tags. Ending
the region at a newline removes the question entirely.

Directives are also *namespaced*: ``!margAI:`` belongs to the framework, and
each domain pack takes one of its own (``!sc:``, ``!hr:``). A message may
carry several. Namespacing is what lets two packs both define ``approve``
without ever meeting, and it is why this parser is namespace-agnostic -- it
reports what it found and lets the registry decide what is meaningful.

For compatibility, a directive that is *not* at the front is still parsed, so
``"make it terse  !margAI: refine"`` keeps working. It is reported through
:func:`find_directives` as a trailing directive rather than silently merged.

This module is just the layer. It provides:

- the parser: :class:`Tag` / :func:`find_directives` / :func:`remove_directive`; and
- :func:`install_bangtags`, which wires one request hook that finds the
  directives, strips them from the prompt, and records the tags with
  ``ctx.set_tags`` -- exactly where the wrapper's event dispatch
  looks (see :meth:`margAI.Wrapper.before`).

The layer is optional and stands apart from the tag handlers: those work with
any tag source. If you don't want ``!margAI:`` syntax, skip
``install_bangtags`` and call ``ctx.set_tags(...)`` from your own request hook
the way you like.

    from margAI import Wrapper, install_bangtags

    app = Wrapper.from_config(load_config())
    install_bangtags(app)           # layer-in !margAI: parsing

    @app.before("refine")
    def refine(ctx, tag):
        ...                         # customize here

**Who is allowed to write a directive.** This layer reads a *control channel*
out of a *user channel*, and that is the whole security surface of the feature:
``!margAI: route=openai/gpt-4o`` in a message is a routing instruction, so
anyone who can choose the message can choose the model and spend the money.
The per-call config overlay is guarded (``margAI.core.configview.safe_overlay``
refuses anything but ``[packs.*]`` and ``[models.*]``); nothing guards this,
because the directive path predates the question and reads from text no
boundary is willing to sanitize on the model's behalf.

So the trust decision belongs to whoever installs the layer, and it splits two
ways:

- **You build the prompt.** LibreChat, a script, an internal tool. The message
  is yours, so ``install_bangtags(app)`` with no allowlist is right. There is
  nothing to defend against.

- **A user you do not control writes it.** A chat product, a shared endpoint, a
  public ``margAI/dynamic``. Then an allowlist is the difference between a
  feature and a billing bug::

      install_bangtags(app, allow=["think", "terse", "refine"])

  Anything else on the directive is dropped with a warning. Note that dropping
  a tag is not a substitute for authenticating the endpoint -- someone who can
  send a request can also ask for the most expensive model you host by naming
  it as the ``model`` field, which no allowlist here covers. See the security
  note in the README.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

from .core.context import STATE_KEY
from .core.tags import RESERVED_NAMESPACE, namespace_key

__all__ = [
    "BANGTAG_NS",
    "STATE_KEY",
    "Directive",
    "Tag",
    "find_directive",
    "find_directives",
    "install_bangtags",
    "remove_directive",
    "strip_directives",
]

logger = logging.getLogger("margAI.bangtag")

BANGTAG_NS = "margAI"

# A directive body runs until the next directive or the end of its line. The
# body is non-greedy with that lookahead so two adjacent directives split
# cleanly instead of the first swallowing the second.
_DIRECTIVE_RE = re.compile(
    r"(?<!\S)!([A-Za-z0-9_-]+)\s*:\s*(?P<tags>[^\n]*?)(?=\s*![A-Za-z0-9_-]+\s*:|[ \t]*(?:\n|$))"
)

# The canonical form: one or more directive lines at the very front, after any
# blank lines. The second group is the prompt, captured so callers can keep it
# verbatim rather than reverse-engineering where the region ended.
_HEAD_RE = re.compile(
    r"\A(?:[ \t]*\n)*(?P<head>(?:[ \t]*![A-Za-z0-9_-]+[ \t]*:[^\n]*(?:\n|\Z))+)(?P<prompt>.*)\Z",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class Tag:
    """A single parsed bangtag.

    ``namespace`` is the directive it came from, and is what makes ``think``
    from ``!margAI:`` and ``think`` from ``!hr:`` different tags. It defaults
    to :data:`BANGTAG_NS` so a hand-built ``Tag("terse")`` keeps working.
    """

    name: str
    value: str | None = None
    namespace: str = BANGTAG_NS

    @property
    def qualified(self) -> str:
        """Canonical lookup key: namespace folded to lower case.

        ``Tag`` keeps ``namespace`` as written so a caller can read back what
        the prompt said; matching folds it, so ``!MARGAI: think`` and
        ``!margAI: think`` are the same tag.
        """
        return f"{namespace_key(self.namespace)}:{self.name}"

    def __str__(self) -> str:
        return f"{self.name}={self.value}" if self.value else self.name


@dataclass(frozen=True, slots=True)
class Directive:
    """One ``!ns: ...`` occurrence: what it was, and what it said."""

    namespace: str
    spans: tuple[str, ...]
    tags: tuple[Tag, ...]
    #: False for a directive that was not in the leading block, so callers can
    #: report the deprecated placement instead of silently accepting it.
    leading: bool = True

    @property
    def text(self) -> str:
        """The first matched span -- the single-namespace view."""
        return self.spans[0] if self.spans else ""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(tag.name for tag in self.tags)


def find_directives(text: str) -> list[Directive]:
    """Every ``!ns: ...`` directive in ``text``, in the order written.

    Prefers the leading block; a directive found elsewhere in the message is
    still returned, marked ``leading=False``, so a misplaced one is reported
    rather than dropped. Same namespace twice merges, leading block first.
    """
    if not text:
        return []

    found: list[Directive] = []
    lead_end = 0
    if head := _HEAD_RE.match(text):
        lead_end = head.end("head")
        found.extend(_parse_region(head.group("head"), leading=True))
    if tail := text[lead_end:]:
        found.extend(_parse_region(tail, leading=False))
    return _merge(found)


def _parse_region(region: str, *, leading: bool) -> list[Directive]:
    out: list[Directive] = []
    for match in _DIRECTIVE_RE.finditer(region):
        # Keep the spelling the user wrote: a tag is data the caller can see
        # back, and matching folds case via namespace_key.
        namespace = match.group(1).strip()
        out.append(
            Directive(
                namespace=namespace,
                spans=(match.group(0),),
                tags=_parse_tags(match.group("tags"), namespace),
                leading=leading,
            )
        )
    return out


def _merge(found: list[Directive]) -> list[Directive]:
    """Collapse repeats of one namespace, keeping the leading block's order."""
    out: list[Directive] = []
    index: dict[str, int] = {}
    for directive in found:
        at = index.get(directive.namespace)
        if at is None:
            index[directive.namespace] = len(out)
            out.append(directive)
        else:
            prior = out[at]
            out[at] = Directive(
                namespace=prior.namespace,
                spans=prior.spans + directive.spans,
                tags=prior.tags + directive.tags,
                leading=prior.leading and directive.leading,
            )
    return out


def find_directive(text: str, namespace: str = BANGTAG_NS) -> tuple[str | None, list[Tag]]:
    """``(directive_text, tags)`` for one namespace, or ``(None, [])``.

    The single-namespace view. :func:`find_directives` is the real parser;
    this stays for callers that only care about one directive's namespace.
    """
    for directive in find_directives(text):
        if namespace_key(directive.namespace) == namespace_key(namespace):
            return directive.text, list(directive.tags)
    return None, []


def _parse_tags(raw: str, namespace: str = BANGTAG_NS) -> tuple[Tag, ...]:
    tags: list[Tag] = []
    for part in raw.split(","):
        for chunk in part.split():
            name, sep, value = chunk.partition("=")
            name = name.strip().lower()
            if not name:
                continue
            tags.append(Tag(name=name, value=value.strip() or None if sep else None, namespace=namespace))
    return tuple(tags)


def remove_directive(text: str, directive: str) -> str:
    """Remove a directive from the prompt (default behavior before upstream)."""
    return strip_directives(text, (directive,))


def strip_directives(text: str, directives: tuple[str, ...] | list[str]) -> str:
    """Remove every matched span, keeping the rest of the message verbatim.

    Last span first, so removing one never shifts the text another span was
    located in.

    Whether a span takes its trailing newline depends on whether it *owns* its
    line. A directive alone on a line leaves a blank line behind if the newline
    stays, so it takes the whole line. A directive written inside a line of
    prose must not, because eating that newline would glue the line's
    remaining words onto whatever followed -- and since each pack parses and
    strips only its own namespace, the next hook to run would read the prompt
    as that namespace's tags. Getting this wrong is how "Review Acme Ltd."
    became three tags.
    """
    for span in reversed([d for d in directives if d]):
        whole_line = re.sub(rf"(?m)^[ \t]*{re.escape(span)}[ \t]*\n?", "", text, count=1)
        # Mid-line: remove the span and its trailing spaces, keep the break.
        inline = re.sub(rf"{re.escape(span)}[ \t]*", "", text, count=1)
        text = whole_line if whole_line != text else inline
    return text.strip()


class _Permit:
    """The trust policy for one installed namespace, readable per request.

    Mutable after installation on purpose. The built-in tag pack installs the
    ``margAI`` namespace itself, before any operator code runs, so an
    ``allow`` that only applied at *first* install would be unreachable for
    exactly the namespace that matters most -- the call would look configured
    and do nothing. Holding the policy in a cell the hook reads per request
    means ``install_bangtags(app, allow=[...])`` works whenever it is called.
    """

    __slots__ = ("allowed",)

    def __init__(self, allowed: set[str] | None) -> None:
        #: ``None`` is "trust the message"; a set is an allowlist; an empty set
        #: admits nothing.
        self.allowed = allowed


# Idempotence guard, keyed per (app, namespace): the bookkeeping stays off the
# host app, and installing a *second* namespace on the same app still works.
_installed: WeakKeyDictionary[Any, dict[str, _Permit]] = WeakKeyDictionary()


def install_bangtags(
    app: Any,
    *,
    namespace: str = BANGTAG_NS,
    state_key: str = STATE_KEY,
    order: int = -100,
    allow: Sequence[str] | None = None,
) -> None:
    """Wire bangtag parsing into ``app``'s request phase (idempotent).

    Adds a request hook that finds ``!{namespace}: ...`` in the last user
    message, strips the directive from the prompt, and records the parsed
    ``Tag``\\ s via ``ctx.set_tags(tags, key=state_key)`` -- the key the
    wrapper's event dispatch reads, so ``@app.before("refine")``-style
    handlers fire.

    The hook runs at ``order``, which defaults to before the wrapper's own
    event dispatch, so tags are ready by the time ``before`` events (and
    therefore every tag handler) run.

    Install one namespace per pack. Each installation only claims its own
    directives, so ``install_bangtags(app, namespace="sc")`` coexists with the
    default ``margAI`` one and neither sees the other's tags.

    ``allow`` is the trust boundary. **None (the default) trusts the message**,
    which is right when the prompt comes from your own application and wrong
    when it comes from a user you do not control -- see the module docstring.
    Pass a sequence to admit only those tags; every tag on a directive that is
    not on the list is dropped with a warning, and the rest of the directive
    still applies. An entry matches a bare tag name (``"route"``) or a
    namespaced one (``"margAI:route"``), so an operator who wants
    ``!margAI: think`` but not ``!margAI: route`` can say exactly that. An empty
    sequence admits nothing, which is the way to keep the parsing (and the
    prompt-stripping) while letting no tag through.
    """
    spelled = namespace or RESERVED_NAMESPACE
    ns = namespace_key(spelled)
    permits = _installed.setdefault(app, {})
    if (permit := permits.get(ns)) is not None:
        # Already parsing this namespace. An allowlist supplied now still
        # counts, and tightening is the only direction allowed: a later plain
        # `install_bangtags(app)` must not quietly reopen a namespace someone
        # has just closed, because a second install is far more likely to be
        # "I forgot this was already set up" than a deliberate unlock.
        if allow is not None:
            permit.allowed = {namespace_key(a) for a in allow}
        return
    permit = _Permit(None if allow is None else {namespace_key(a) for a in allow})
    permits[ns] = permit

    @app.before(name=f"bangtags:{spelled}", order=order)
    def parse_bangtags(ctx: Any) -> None:  # pragma: no cover - trivial
        text = _last_user_message(ctx)
        if not text:
            ctx.set_tags((), key=state_key, namespace=spelled)
            return
        directives = [d for d in find_directives(text) if namespace_key(d.namespace) == ns]
        # Strip before filtering, and strip every span we matched. A directive
        # whose tags were all denied still gets removed from the prompt: the
        # text is the framework's to consume either way, and leaving it behind
        # would hand the caller the one thing the allowlist just refused --
        # a `!margAI: route=openai/gpt-4o` sitting in the model's context,
        # where an instruction-following model may well act on it.
        if directives:
            _strip_directive(ctx, tuple(s for d in directives for s in d.spans))
        if permit.allowed is not None:
            directives = [_filter_tags(d, permit.allowed, spelled) for d in directives]
        directives = [d for d in directives if d.tags]
        # Namespace-scoped so several installed packs can share one tag list;
        # each hook contributes its own directive and keeps the others'.
        # `key` is passed too because set_tags points ctx.tags at it, so a
        # non-default state_key is actually read back by the event dispatch.
        ctx.set_tags([tag for d in directives for tag in d.tags], key=state_key, namespace=spelled)
        ctx.state.setdefault("margAI", {})["directive"] = directives[0].text if directives else None


def _filter_tags(directive: Directive, permitted: set[str], namespace: str) -> Directive:
    """Drop the tags of one directive that the allowlist does not admit.

    A rejected tag is dropped rather than failing the call: a directive that
    mixes an allowed tag with a denied one is an ordinary prompt, and a
    multi-tenant deployment should degrade to the safe subset instead of
    turning a stray ``!margAI:`` in quoted text into a 400.
    """
    kept = tuple(t for t in directive.tags if t.qualified in permitted or t.name in permitted)
    dropped = [t for t in directive.tags if t not in kept]
    if dropped:
        # Warning, not debug: a tag the operator configured away being set from
        # a message is either a misconfiguration or an attempt, and both are
        # things an operator wants to see in a log they already read.
        logger.warning(
            "dropped bangtag(s) not on the allowlist for namespace '%s': %s",
            namespace,
            ", ".join(sorted({t.name for t in dropped})),
        )
    return Directive(
        namespace=directive.namespace,
        spans=directive.spans,
        tags=kept,
        leading=directive.leading,
    )


def _last_user_message(ctx: Any) -> str | None:
    for msg in reversed(ctx.messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    return None


def _strip_directive(ctx: Any, spans: tuple[str, ...]) -> None:
    for msg in reversed(ctx.messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            msg["content"] = strip_directives(msg["content"], spans)
            return
