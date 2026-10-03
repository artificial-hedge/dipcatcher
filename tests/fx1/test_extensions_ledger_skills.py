"""Generated skill declaration shards reach the public extension registry.

``scripts/generated_capability_declarations/skills/<owner>.py`` is the
expanded, one-call-per-seed form of the compact seed layout in
:mod:`fx1.capabilities`. Each shard names its runtime wrapper
(``fx1.extensions.skills.<owner>.MODULE``) and registers the same seed ids.
Until the compact seed runtime existed, importing a shard -- and therefore
every skill wrapper in :mod:`fx1.extensions` -- raised ``ModuleNotFoundError``,
so the whole skill path was unreachable. These tests drive the skill shards
through their owner and back out of the public ``fx1.extensions`` API.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from fx1.capabilities import resolve_seed_id
from fx1.extensions import extension_manifest, get_extension

_OWNERS = (
    "candle-book",
    "collect",
    "doctor",
    "forecast",
    "ingest",
    "kronos-forecast",
    "kyle-ofi",
    "monitor",
    "northset",
    "optimize",
    "paper",
    "report",
    "research",
    "session-book",
)

_LEDGER = (
    Path(__file__).resolve().parents[2] / "scripts" / "generated_capability_declarations" / "skills"
)


def _load_shard(owner: str) -> ModuleType:
    """Import one generated declaration shard by path, not by package name."""
    module_name = "_capability_ledger_skill_" + owner.replace("-", "_")
    path = _LEDGER / (owner.replace("-", "_") + ".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    shard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shard)
    return shard


@pytest.mark.parametrize("owner", _OWNERS)
def test_declaration_shard_matches_runtime_wrapper(owner: str) -> None:
    shard = _load_shard(owner)
    wrapper = get_extension("skill", owner)

    assert shard.extension() is wrapper
    assert tuple(shard.CAPABILITY_REFERENCES) == tuple(wrapper.references)
    assert wrapper.card_count == len(wrapper.references) > 0


@pytest.mark.parametrize("owner", _OWNERS)
def test_registered_skill_resolves_back_to_its_owner(owner: str) -> None:
    wrapper = get_extension("skill", owner)

    first = resolve_seed_id(int(wrapper.references[0]))
    last = resolve_seed_id(int(wrapper.references[-1]))

    assert first["kind"] == "skill"
    assert first["command"] == owner
    assert last["kind"] == "skill"
    assert last["command"] == owner


@pytest.mark.parametrize("owner", _OWNERS)
def test_registered_skill_is_reachable_through_manifest(owner: str) -> None:
    manifest = extension_manifest("skill", owner)

    assert manifest["schema"] == "fx1.extension-module/v1"
    assert manifest["kind"] == "skill"
    assert manifest["command"] == owner
    assert manifest["market_evidence"] is False
    assert manifest["operations"] == ["run"]


def test_unknown_skill_fails_closed() -> None:
    with pytest.raises(KeyError):
        get_extension("skill", "not-a-registered-skill")
