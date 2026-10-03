"""The feature declaration ledger reaches the public extension registry.

Slice ``fx-003`` owns the feature shards ``rel_volume`` through ``vol_ewma``.
Each generated shard under ``scripts/generated_capability_declarations/features/``
names its runtime wrapper (``fx1.extensions.features.<owner>.MODULE``) and
registers the same seed ids. Every wrapper imports ``fx1.capabilities``, which
did not exist, so the shards and the ``fx1.extensions`` feature API were
unreachable (``ModuleNotFoundError``). These tests drive each owned shard
through its owner and back out of the public ``fx1.extensions`` surface.

The seed ids are generated discovery records, never market evidence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from fx1.capabilities import resolve_seed_id
from fx1.extensions import extension_manifest, get_extension
from fx1.extensions.feature_catalog import get_feature_definition, list_feature_definitions

_OWNERS = (
    "rel_volume",
    "ret_1",
    "ret_20",
    "ret_5",
    "ret_intraday_20",
    "ret_open_close",
    "ret_overnight",
    "ret_overnight_20",
    "reversal_1",
    "skew_20",
    "turnover_proxy",
    "vol_20",
    "vol_60",
    "vol_ewma",
)

_LEDGER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "generated_capability_declarations"
    / "features"
)


def _load_shard(owner: str) -> ModuleType:
    """Import one generated declaration shard by path, not by package name."""
    path = _LEDGER / f"{owner}.py"
    spec = importlib.util.spec_from_file_location(f"_capability_ledger_{owner}", path)
    assert spec is not None and spec.loader is not None
    shard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shard)
    return shard


def test_owned_feature_shards_exist() -> None:
    for owner in _OWNERS:
        assert (_LEDGER / f"{owner}.py").is_file(), owner


def test_owned_features_are_in_catalog() -> None:
    catalog = {feature.name for feature in list_feature_definitions()}
    assert set(_OWNERS) <= catalog


@pytest.mark.parametrize("owner", _OWNERS)
def test_declaration_shard_matches_runtime_wrapper(owner: str) -> None:
    shard = _load_shard(owner)
    wrapper = get_extension("feature", owner)

    assert shard.extension() is wrapper
    assert tuple(shard.CAPABILITY_REFERENCES) == tuple(wrapper.references)
    assert wrapper.card_count == len(wrapper.references) > 0


@pytest.mark.parametrize("owner", _OWNERS)
def test_registered_feature_is_reachable_through_manifest(owner: str) -> None:
    manifest = extension_manifest("feature", owner)

    assert manifest["schema"] == "fx1.extension-module/v1"
    assert manifest["kind"] == "feature"
    assert manifest["feature"] == owner
    assert manifest["market_evidence"] is False
    assert manifest["operations"] == ["metadata", "build"]


@pytest.mark.parametrize("owner", _OWNERS)
def test_wrapper_metadata_matches_catalog(owner: str) -> None:
    wrapper = get_extension("feature", owner)

    assert wrapper.metadata() == get_feature_definition(owner)
    assert wrapper.metadata().synthetic_only is False


@pytest.mark.parametrize("owner", _OWNERS)
def test_every_seed_id_resolves_back_to_owner(owner: str) -> None:
    wrapper = get_extension("feature", owner)

    wrapper.verify()
    assert resolve_seed_id(wrapper.references[0])["feature"] == owner
    assert resolve_seed_id(wrapper.references[-1])["feature"] == owner


def test_unknown_feature_fails_closed() -> None:
    with pytest.raises(KeyError):
        get_extension("feature", "not_a_registered_feature")
