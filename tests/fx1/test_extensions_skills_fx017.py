"""The generated skill ledger reaches the public extension registry.

scripts/generated_capability_declarations/skills/<owner>.py is the expanded,
one-call-per-seed form of the compact seed layout in fx1.capabilities. Until
that compact runtime existed, importing any skill wrapper in
fx1.extensions.skills raised ModuleNotFoundError and the whole skill path was
unreachable through fx1.extensions.

These tests drive the declaration shards through their owner and back out of
the public fx1.extensions API, and pin the fail-closed harness binding.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from fx1.extensions import extension_manifest, get_extension
from fx1.extensions.contracts import SkillExtension
from fx1.harness import Harness

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
)

_LEDGER = (
    Path(__file__).resolve().parents[2] / "scripts" / "generated_capability_declarations" / "skills"
)


def _shard_path(owner: str) -> Path:
    stem = owner.replace("-", "_")
    return _LEDGER / (stem + ".py")


def _load_shard(owner: str) -> ModuleType:
    """Import one generated declaration shard by path, not by package name."""
    path = _shard_path(owner)
    name = f"_capability_ledger_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    shard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shard)
    return shard


@pytest.mark.parametrize("owner", _OWNERS)
def test_declaration_shard_matches_skill_wrapper(owner: str) -> None:
    shard = _load_shard(owner)
    wrapper = get_extension("skill", owner)

    assert isinstance(wrapper, SkillExtension)
    assert shard.extension() is wrapper
    assert tuple(shard.CAPABILITY_REFERENCES) == tuple(wrapper.references)
    assert wrapper.card_count == len(wrapper.references) > 0
    wrapper.verify()


@pytest.mark.parametrize("owner", _OWNERS)
def test_skill_binds_only_its_registered_harness_command(owner: str) -> None:
    command = get_extension("skill", owner).command()

    assert command.name == owner
    assert command.argv == [owner]
    assert [c.name for c in Harness().list_commands() if c.name == owner] == [owner]


@pytest.mark.parametrize("owner", _OWNERS)
def test_skill_reachable_through_manifest(owner: str) -> None:
    manifest = extension_manifest("skill", owner)

    assert manifest["schema"] == "fx1.extension-module/v1"
    assert manifest["kind"] == "skill"
    assert manifest["command"] == owner
    assert manifest["market_evidence"] is False
    assert manifest["operations"] == ["run"]


@pytest.mark.parametrize("owner", _OWNERS)
def test_skill_run_routes_through_injected_harness(owner: str) -> None:
    calls: list[list[str]] = []

    def recorder(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        calls.append(list(argv))
        return 0, "ok", ""

    result = get_extension("skill", owner).run(harness=Harness(runner=recorder))

    assert calls == [[owner]]
    assert result.command == owner
    assert result.ok


def test_unregistered_skill_fails_closed() -> None:
    with pytest.raises(KeyError):
        get_extension("skill", "not-a-registered-skill")
