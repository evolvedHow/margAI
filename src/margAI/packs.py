"""Finding packs the way pip finds them.

A pack that has to be imported by hand -- ``import my_pack; my_pack.install(app)``
somewhere in the user's ``main.py`` -- is one more thing to forget, and a
forgotten pack fails silently as "my tag does nothing" rather than as an
error. The ``margAI.packs`` entry point group moves that decision to install
time, where ``pip`` already owns the bookkeeping::

    [project.entry-points."margAI.packs"]
    supply-chain = "supply_chain:install"

The contract is deliberately the whole of the API: an entry point resolves to
one callable taking the wrapper. A pack that needs more than that is a pack
that wants a marglet, and should register one.

Failure policy is the part worth stating. An entry point is discovered from
*whatever is installed*, not from a list the operator wrote, so a broken
third-party package must not stop the gateway from starting: the alternative
is that installing an unrelated dependency takes production down with a
traceback about someone else's code. A pack that fails to install is logged
at ERROR, recorded on the wrapper, and skipped; ``margAI doctor`` then reports
it by name. A pack the operator asked for *by name* in
``[packs] only = [...]`` is a different thing, and does raise.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Protocol

__all__ = [
    "ENTRY_POINT_GROUP",
    "EntryPointLike",
    "PackFailure",
    "PackRecord",
    "discover",
    "install_all",
]

logger = logging.getLogger("margAI.packs")

ENTRY_POINT_GROUP = "margAI.packs"


@dataclass(frozen=True)
class _CoreEntryPoint:
    """The built-in pack, shaped like an entry point so it installs by the same
    code path. A real ``importlib.metadata.EntryPoint`` would need the
    distribution to be installed before its metadata could name it, which is
    exactly the thing that is not true while the package is being used from a
    source checkout."""

    name: str
    value: str = "margAI.builtin_tags:install"

    def load(self) -> Any:
        from .builtin_tags import install

        return install


def _core_entry_point() -> EntryPointLike:
    from .builtin_tags import CORE_PACK

    return _CoreEntryPoint(name=CORE_PACK)


class EntryPointLike(Protocol):
    """The part of an entry point :func:`install_all` actually uses.

    A protocol rather than ``importlib.metadata.EntryPoint`` because that is a
    concrete class, and testing the discovery path should not require
    installing a distribution to get a valid instance of it.
    """

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> str: ...

    def load(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class PackFailure:
    """A discovered pack that would not install."""

    name: str
    reason: str
    target: str = ""

    def describe(self) -> str:
        where = f" ({self.target})" if self.target else ""
        return f"{self.name}{where}: {self.reason}"


@dataclass(slots=True)
class PackRecord:
    """What happened to one pack, for ``/v1/margAI/tags`` and the doctor."""

    name: str
    target: str
    installed: bool
    failure: PackFailure | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


def discover(group: str = ENTRY_POINT_GROUP) -> list[EntryPointLike]:
    """Every pack entry point, unsorted.

    The order is decided by :func:`install_all`, not here, so the guarantee
    holds for every source of entry points rather than only the scan.
    """
    try:
        return sorted(entry_points(group=group), key=lambda ep: ep.name)
    except Exception:  # a broken distribution's metadata can break the scan
        logger.exception("cannot scan the %s entry point group", group)
        return []


def _resolve(ep: EntryPointLike) -> Any:
    loaded = ep.load()
    if not callable(loaded):
        raise TypeError(f"entry point is {type(loaded).__name__}, expected a callable")
    return loaded


def _install_one(app: Any, ep: EntryPointLike, *, wanted: set[str] | None) -> PackRecord:
    """Install one entry point, recording what it claimed or why it failed.

    A pack named in ``only`` is a request, so its failure raises; one merely
    discovered is logged and recorded, and ``margAI doctor`` reports it by name.
    """
    target = getattr(ep, "value", "") or ""
    try:
        before = _tag_names(app)
        _resolve(ep)(app)
        after = _tag_names(app)
        return PackRecord(
            name=ep.name,
            target=target,
            installed=True,
            tags=tuple(sorted(after - before)),
        )
    except Exception as exc:
        if wanted is not None:
            raise
        reason = f"{type(exc).__name__}: {exc}"
        logger.error("pack %s failed to install: %s", ep.name, reason, exc_info=True)
        return PackRecord(
            name=ep.name,
            target=target,
            installed=False,
            failure=PackFailure(ep.name, reason, target),
        )


def _disabled_by_config(app: Any, skip: Sequence[str]) -> set[str]:
    """Packs the operator has opted out of, by ``skip`` or ``[packs.*]``."""
    disabled = set(skip)
    settings = getattr(getattr(app, "config", None), "packs", None) or {}
    for name, table in settings.items():
        if isinstance(table, dict) and table.get("enabled") is False:
            disabled.add(name)
    return disabled


def install_all(
    app: Any,
    *,
    group: str = ENTRY_POINT_GROUP,
    only: Sequence[str] | None = None,
    skip: Sequence[str] = (),
    enabled: bool = True,
    eps: Sequence[EntryPointLike] | None = None,
) -> list[PackRecord]:
    """Discover and install packs onto ``app``.

    ``only`` restricts installation to named packs, and those failures *do*
    raise: naming a pack is a request, and answering it with a log line is how
    "I enabled the pack and nothing happened" happens. ``skip`` and the
    ``[packs.<name>] enabled = false`` table are the opposite -- opting out
    should never be a thing that can fail.

    The built-in pack ships inside this distribution, so it is not something the
    entry point group can report. It is installed here instead, and it installs
    *first*: tag precedence breaks ties by registration order, and application
    tags should be able to displace a built-in one on purpose. Its installation
    is independent of ``enabled``, which governs third-party *discovery* --
    otherwise ``pack_discovery = false`` would silently take away the
    ``route=`` and ``provider=`` directives the documentation calls zero-config.
    Opt it out by name with ``[packs."margAI.builtin_tags"] enabled = false``.

    ``eps`` exists so the discovery path can be tested without installing
    anything into the environment, and so a caller can install an explicit set
    in place of the built-in pack.
    """
    records: list[PackRecord] = []
    wanted = set(only) if only is not None else None
    disabled = _disabled_by_config(app, skip)

    if eps is None:
        from .builtin_tags import CORE_PACK

        if (wanted is None or CORE_PACK in wanted) and CORE_PACK not in disabled:
            records.append(_install_one(app, _core_entry_point(), wanted=wanted))

    if not enabled:
        return records

    # Sorted here, not at the source: installation order is observable -- a
    # tag's handler order breaks ties by registration -- and a gateway whose
    # tag precedence changed when an unrelated package was upgraded would be a
    # genuinely miserable bug to chase.
    for ep in sorted(eps if eps is not None else discover(group), key=lambda e: e.name):
        if wanted is not None and ep.name not in wanted:
            continue
        if ep.name in disabled:
            logger.debug("pack %s is disabled by configuration", ep.name)
            continue
        records.append(_install_one(app, ep, wanted=wanted))
    return records


def _tag_names(app: Any) -> set[str]:
    """Qualified tag names currently registered.

    Diffed around the install call to report what a pack actually claimed. The
    claim is the useful part: "installed: yes" proves a callable returned,
    which is a much weaker statement than "it registered the tag you expected".
    """
    registry = getattr(app, "tags", None)
    if registry is None:
        return set()
    try:
        return {f"{ns}:{entry['name']}" for ns, entries in registry.describe().items() for entry in entries}
    except Exception:
        return set()


def describe(records: Sequence[PackRecord]) -> Iterator[dict[str, Any]]:
    for record in records:
        yield {
            "name": record.name,
            "target": record.target,
            "installed": record.installed,
            "tags": list(record.tags),
            "error": record.failure.reason if record.failure else None,
        }
