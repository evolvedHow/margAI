"""Tag handlers: the one mechanism by which a tag does anything.

A tag on a call is inert data until something claims it. This module holds
that claim -- who registered a handler for which tag, in which phase, owned by
which pack -- and nothing else. It deliberately knows *no tag names*.

The rule the rest of the framework is built on:

    **The framework orchestrates the pipeline. Handlers decide behaviour.**

So there is no ``if name == "route"`` anywhere in the orchestration layer, not
even for the tags that ship with margAI. ``!margAI: route=`` works because a
handler in :mod:`margAI.builtin_tags` says so, registered through the same
decorator a third-party pack uses. The framework's only job is to parse tags,
find the handlers that claim them, and call them in a defined order.

Namespacing is what makes plug-in safe. Each pack owns one directive
(``!sc:``, ``!hr:``), so two domains can both define ``approve`` without ever
meeting, and a directive says who owns what. ``margAI`` is reserved for the
framework, and a pack cannot take it.

Ownership is enforced, not documented: :meth:`TagRegistry.register` refuses a
replacement from any pack other than the root, so "no packs can be overridden"
is a property of the data structure rather than a convention.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CORE_PACKS",
    "RESERVED_NAMESPACE",
    "ROOT_PACK",
    "TagConflictError",
    "TagHandler",
    "TagRegistry",
    "namespace_key",
    "qualified_key",
    "qualify",
]

logger = logging.getLogger("margAI.tags")

#: Owner recorded for handlers registered outside any ``app.pack(...)`` block:
#: the application's own code, which is the only thing allowed to replace a
#: handler someone else registered.
ROOT_PACK = "<app>"

#: The namespace the framework's own tags live in. A pack may not claim it.
RESERVED_NAMESPACE = "margAI"

#: Packs allowed to register into :data:`RESERVED_NAMESPACE`: the application
#: authoring its own deployment, and the built-in pack the framework ships as
#: a pack like any other. Everything else needs its own namespace.
CORE_PACKS = frozenset({ROOT_PACK, "margAI.builtin_tags"})


class TagConflictError(Exception):
    """A pack tried to take a tag, namespace, or handler it does not own."""


def namespace_key(namespace: str) -> str:
    """The identity of a namespace: case- and whitespace-insensitive.

    A namespace is a name, and nobody should get "unknown tag" for typing
    ``!MARGAI:`` instead of ``!margAI:``. So matching folds case, while
    :attr:`TagRegistry.describe` keeps the spelling the pack registered with --
    the brand stays ``margAI`` even though the key behind it is not.
    """
    return namespace.strip().lower()


def qualified_key(name: str, namespace: str = RESERVED_NAMESPACE) -> str:
    """The canonical registry key for a tag: folded namespace, verbatim name.

    Every producer of a lookup key goes through here -- :func:`qualify` for
    tags arriving on a call, :meth:`TagRegistry.register` for handlers going
    in -- so the two sides of a match cannot disagree about case.
    """
    return f"{namespace_key(namespace)}:{name}"


def qualify(tag: Any) -> str:
    """``"namespace:name"`` for anything carrying a name.

    Tags are duck-typed -- the pipeline only ever requires a ``.name`` -- so a
    foreign tag with no namespace degrades to its bare name instead of
    raising. That keeps hand-built tags working without adopting this class.
    """
    qualified = getattr(tag, "qualified", None)
    if isinstance(qualified, str) and qualified:
        return qualified
    name = getattr(tag, "name", None)
    if not name:
        return ""
    namespace = getattr(tag, "namespace", "") or ""
    return qualified_key(str(name), namespace) if namespace else str(name)


@dataclass(frozen=True, slots=True)
class TagHandler:
    """One registered claim on one tag, in one phase."""

    name: str
    namespace: str
    phase: str
    fn: Callable[..., Any]
    pack: str = ROOT_PACK
    order: int = 0
    seq: int = 0
    doc: str = ""

    @property
    def qualified(self) -> str:
        """Canonical, case-folded lookup key -- equal to :func:`qualify`."""
        return qualified_key(self.name, self.namespace)

    def summary(self) -> str:
        """First doc line -- what a newcomer sees in ``describe()``."""
        for source in (inspect.getdoc(self.fn), self.doc):
            if source:
                return source.strip().splitlines()[0]
        return ""


class TagRegistry:
    """Tag -> handlers, with pack ownership and deterministic ordering.

    Handlers for one tag accumulate rather than overwrite, because two handlers
    claiming ``!margAI: think`` -- one steering routing, one setting
    ``reasoning_effort`` -- is composition, and composition is the point. What
    is refused is one pack *removing* another's handler; that is the
    difference between adding a behaviour and overwriting one.
    """

    def __init__(self, namespace: str = RESERVED_NAMESPACE) -> None:
        # (qualified, phase) -> handlers, in registration order. Keys are
        # canonical: namespace folded to lower case.
        self._by_tag: dict[tuple[str, str], list[TagHandler]] = {}
        # canonical namespace -> the pack that claimed it.
        self._owner: dict[str, str] = {}
        # canonical namespace -> the spelling a pack registered it under, so
        # introspection reports `margAI`, not `margai`.
        self._display: dict[str, str] = {}
        self._seq = 0
        # The application's own namespace, reserved from the start rather than
        # left to whichever pack happened to register first. Configurable
        # because a deployment that renames its model prefix wants its
        # directives renamed with it; the framework's built-in pack still
        # registers inside `margAI` explicitly, so the two coexist.
        self.reserved_namespace = namespace or RESERVED_NAMESPACE
        self._reserved_key = namespace_key(self.reserved_namespace)
        self._owner[self._reserved_key] = ROOT_PACK
        self._display[self._reserved_key] = self.reserved_namespace

    @property
    def reserved(self) -> str:
        """The application's own namespace, as spelled at construction."""
        return self.reserved_namespace

    # -- namespaces ---------------------------------------------------------

    def reserve(self, namespace: str, pack: str) -> None:
        """Claim ``namespace`` for ``pack``.

        Idempotent for the same pack, so a reload is not a conflict. A second
        *pack* claiming it is refused, which is what stops a domain pack from
        quietly taking over ``margAI`` or another domain's directive.

        The application's own code is the one exception. ``<app>`` is not a
        pack -- it is whoever is authoring the deployment -- so it may take a
        namespace over, loudly. That is the deliberate-override escape hatch,
        and a log line is the price of it: overriding another pack's directive
        should never be something you find out about from a support ticket.
        """
        key = namespace_key(namespace)
        owner = self._owner.get(key)
        if owner is not None and owner != pack:
            if not (key == self._reserved_key and pack in CORE_PACKS):
                if pack != ROOT_PACK:
                    raise TagConflictError(
                        f"namespace {self._display.get(key, key)!r} is owned by pack {owner!r}; "
                        f"pack {pack!r} may not claim it"
                    )
                logger.warning(
                    "application code claiming namespace %r, previously owned by pack %r",
                    self._display.get(key, key),
                    owner,
                )
            else:
                # Framework code registering inside the application's namespace.
                # It is a guest there, not the owner, so a later `app.pack(...)`
                # on `margAI` is still the application's to take.
                self._display.setdefault(key, namespace)
                return
        self._owner[key] = pack
        self._display.setdefault(key, namespace)

    def owner(self, namespace: str) -> str | None:
        return self._owner.get(namespace_key(namespace))

    def namespaces(self) -> set[str]:
        """Namespaces as their owners spelled them."""
        return {self._display.get(key, key) for key in self._owner}

    # -- registration -------------------------------------------------------

    def register(
        self,
        *,
        name: str,
        namespace: str,
        phase: str,
        fn: Callable[..., Any],
        pack: str = ROOT_PACK,
        order: int = 0,
        replace: bool = False,
    ) -> TagHandler:
        if not name:
            raise TagConflictError("a tag handler needs a name")
        if namespace_key(namespace) == self._reserved_key and pack not in CORE_PACKS:
            raise TagConflictError(
                f"namespace {self.reserved_namespace!r} is reserved for the framework; "
                f"pack {pack!r} may not register into it"
            )
        self.reserve(namespace, pack)

        key = (qualified_key(name, namespace), phase)
        existing = self._by_tag.get(key, [])

        if replace and existing:
            # "No packs can be overridden." Only the application's own code --
            # a deliberate act while authoring a deployment -- may displace a
            # handler another pack registered. A pack replacing its own
            # handler is a reload, which is why the owner is exempt.
            foreign = [h for h in existing if h.pack != pack]
            if foreign and pack != ROOT_PACK:
                owners = ", ".join(sorted({h.pack for h in foreign}))
                raise TagConflictError(
                    f"pack {pack!r} may not replace the handler(s) for "
                    f"{key[0]!r} owned by {owners}; register your own tag, or "
                    f"redefine it from application code"
                )
            if foreign:
                logger.warning(
                    "application code replacing %s handler(s) for %r (was %s)",
                    len(foreign),
                    key[0],
                    ", ".join(sorted({h.pack for h in foreign})),
                )
            self._by_tag[key] = []

        handler = TagHandler(
            name=name,
            namespace=namespace,
            phase=phase,
            fn=fn,
            pack=pack,
            order=order,
            seq=self._seq,
        )
        self._seq += 1
        self._by_tag.setdefault(key, []).append(handler)
        return handler

    # -- lookup -------------------------------------------------------------

    def handlers(self, qualified: str, phase: str) -> list[TagHandler]:
        """Handlers claiming ``qualified`` in ``phase``, in execution order."""
        found = self._by_tag.get((qualified, phase), [])
        return sorted(found, key=lambda h: (h.order, h.seq))

    def plan(self, tags: Any, phase: str) -> list[tuple[TagHandler, Any]]:
        """The dispatch plan for one phase: ``(handler, tag)`` pairs.

        Tags are walked in the order given, and each tag's handlers run in
        ``(order, seq)``. Tag order is therefore the caller's to decide --
        the wrapper resolves it once, in ``_dispatch_tags``, and hands the
        answer down here so there is exactly one place that reorders.

        Pairing each handler with the concrete tag the caller wrote is what
        keeps ``fn(ctx, tag)`` honest when several tags map to one handler.
        """
        pairs: list[tuple[TagHandler, Any]] = []
        for tag in tags:
            qualified = qualify(tag)
            if not qualified:
                continue
            group = self._by_tag.get((qualified, phase), [])
            pairs.extend((handler, tag) for handler in sorted(group, key=lambda h: (h.order, h.seq)))
        return pairs

    def handlers_for(self, tag: Any) -> list[TagHandler]:
        """Every handler for ``tag`` across all phases, in execution order."""
        qualified = qualify(tag)
        found = [h for (q, _), group in self._by_tag.items() if q == qualified for h in group]
        return sorted(found, key=lambda h: (h.phase, h.order, h.seq))

    def known(self) -> set[str]:
        """Qualified names that have at least one handler."""
        return {qualified for qualified, _ in self._by_tag}

    def bare_names(self) -> set[str]:
        """Unqualified tag names, for back-compat with the pre-namespace API."""
        return {handler.name for group in self._by_tag.values() for handler in group}

    def owner_of_tag(self, qualified: str) -> str | None:
        namespace = qualified.rsplit(":", 1)[0] if ":" in qualified else ""
        return self._owner.get(namespace_key(namespace))

    def __len__(self) -> int:
        return sum(len(group) for group in self._by_tag.values())

    def __contains__(self, qualified: object) -> bool:
        return qualified in self.known()

    # -- introspection ------------------------------------------------------

    def describe(self) -> dict[str, list[dict[str, Any]]]:
        """The full registry, grouped by namespace.

        This is what a newcomer asks for -- "what can this gateway's tags
        actually do?" -- and what the error messages and ``margAI doctor``
        read, so it can never drift from what is really registered.
        """
        out: dict[str, list[dict[str, Any]]] = {}
        for (qualified, _phase), group in sorted(self._by_tag.items()):
            key, _, name = qualified.rpartition(":")
            namespace = self._display.get(key, key)
            out.setdefault(namespace, []).append(
                {
                    "name": name,
                    "phases": sorted({h.phase for h in group}),
                    "packs": sorted({h.pack for h in group}),
                    "doc": group[0].summary(),
                }
            )
        return out

    def docs(self) -> dict[str, str]:
        """Back-compat view: bare tag name -> first doc line."""
        out: dict[str, str] = {}
        for group in self._by_tag.values():
            for handler in group:
                out.setdefault(handler.name, handler.summary())
        return out
