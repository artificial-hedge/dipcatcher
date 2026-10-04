"""Plugin and skill declaration shards reach the public extension registry.

``scripts/generated_capability_declarations/plugins/<owner>.py`` and
``.../skills/<owner>.py`` are the expanded, one-call-per-seed form of the
compact seed layout in :mod:`fx1.capabilities`. Each shard names its runtime
wrapper (``fx1.extensions.plugins.<owner>.MODULE`` /
``fx1.extensions.skills.<owner>.MODULE``) and registers the same seed ids.

Before the compact seed runtime existed the wrappers imported a nonexistent
``fx1.capabilities`` module, so every declaration shard -- and therefore every
registered plugin/skill extension -- raised ``ModuleNotFoundError`` and was
unreachable. These tests drive this slices shards through their owner and
back out of the public ``fx1.extensions`` API.

The seed ids are generated discovery records, not independently verified
capabilities and never market evidence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from fx1.extensions import extension_manifest, get_extension, list_extensions

# (ledger file stem, registered owner) -- the ledger basename is underscored,
# the harness command name is hyphenated.
_PLUGIN_OWNERS = (
    ("ifind", "ifind"),
    ("igo_open_data", "igo_open_data"),
    ("imf", "imf"),
    ("sec_edgar", "sec_edgar"),
    ("sp_data", "sp_data"),
    ("tianyancha", "tianyancha"),
    ("wind", "wind"),
    ("world_bank", "world_bank"),
    ("xhcj", "xhcj"),
    ("yahoo_finance", "yahoo_finance"),
)

_SKILL_OWNERS = (
    ("backtest", "backtest"),
    ("book_panel", "book-panel"),
    ("build_features", "build-features"),
    ("build_labels", "build-labels"),
)

_LEDGER = Path(__file__).resolve().parents[2] / "scripts" / "generated_capability_declarations"


def _load_shard(kind: str, stem: str) -> ModuleType:
    """Import one generated declaration shard by path, not by package name."""
    path = _LEDGER / kind / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_capability_ledger_{kind}_{stem}", path)
    assert spec is not None and spec.loader is not None
    shard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shard)
    return shard


@pytest.mark.parametrize("stem,owner", _PLUGIN_OWNERS)
def test_plugin_shard_matches_runtime_wrapper(stem: str, owner: str) -> None:
    shard = _load_shard("plugins", stem)
    wrapper = get_extension("plugin", owner)

    assert shard.extension() is wrapper
    assert tuple(shard.CAPABILITY_REFERENCES) == tuple(wrapper.references)
    assert wrapper.card_count == len(wrapper.references) > 0


@pytest.mark.parametrize("stem,owner", _SKILL_OWNERS)
def test_skill_shard_matches_runtime_wrapper(stem: str, owner: str) -> None:
    shard = _load_shard("skills", stem)
    wrapper = get_extension("skill", owner)

    assert shard.extension() is wrapper
    assert tuple(shard.CAPABILITY_REFERENCES) == tuple(wrapper.references)
    assert wrapper.card_count == len(wrapper.references) > 0


@pytest.mark.parametrize("stem,owner", _PLUGIN_OWNERS)
def test_registered_plugin_is_reachable_through_manifest(stem: str, owner: str) -> None:
    manifest = extension_manifest("plugin", owner)

    assert manifest["schema"] == "fx1.extension-module/v1"
    assert manifest["kind"] == "plugin"
    assert manifest["source"] == owner
    assert manifest["market_evidence"] is False
    assert manifest["operations"] == ["probe", "describe", "fetch"]


@pytest.mark.parametrize("stem,owner", _SKILL_OWNERS)
def test_registered_skill_is_reachable_through_manifest(stem: str, owner: str) -> None:
    manifest = extension_manifest("skill", owner)

    assert manifest["schema"] == "fx1.extension-module/v1"
    assert manifest["kind"] == "skill"
    assert manifest["command"] == owner
    assert manifest["market_evidence"] is False
    assert manifest["operations"] == ["run"]


def test_list_extensions_exposes_slice_owners() -> None:
    plugin_owners = {entry["owner"] for entry in list_extensions("plugin")}
    skill_owners = {entry["owner"] for entry in list_extensions("skill")}

    assert {owner for _, owner in _PLUGIN_OWNERS} <= plugin_owners
    assert {owner for _, owner in _SKILL_OWNERS} <= skill_owners


def test_unknown_plugin_and_skill_fail_closed() -> None:
    with pytest.raises(KeyError):
        get_extension("plugin", "not_a_registered_plugin")
    with pytest.raises(KeyError):
        get_extension("skill", "not-a-registered-skill")
