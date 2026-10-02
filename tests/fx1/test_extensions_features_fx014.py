"""The fx-014 feature wrappers reach the public extension registry.

Before ``src/fx1/capabilities.py`` existed, every wrapper under
``src/fx1/extensions/features/`` imported ``fx1.capabilities`` at module top
level, so ``fx1.extensions.get_extension("feature", ...)`` raised
``ModuleNotFoundError`` for all 46 registered feature owners.

This slice owns 14 of those wrappers (``ret_open_close`` through
``volume_vol``). These tests drive each one through the public
``fx1.extensions`` API and cross-check it against its generated declaration
shard under ``scripts/generated_capability_declarations/features/``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from fx1.capabilities import owner_references, resolve_seed_id
from fx1.extensions import extension_manifest, get_extension, list_extensions
from fx1.extensions.feature_catalog import get_feature_definition

_OWNERS = (
    "ret_open_close",
    "ret_overnight",
    "ret_overnight_20",
    "reversal_1",
    "skew_20",
    "turnover_proxy",
    "vol_20",
    "vol_60",
    "vol_ewma",
    "vol_garman_klass",
    "vol_of_vol",
    "vol_parkinson",
    "vol_ratio_20_60",
    "volume_vol",
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


@pytest.mark.parametrize("owner", _OWNERS)
def test_wrapper_is_registered_and_reachable(owner: str) -> None:
    wrapper = get_extension("feature", owner)

    assert wrapper.kind == "feature"
    assert wrapper.owner == owner
    assert wrapper.module == f"fx1.extensions.features.{owner}"
    assert wrapper.card_count == len(wrapper.references) > 0


@pytest.mark.parametrize("owner", _OWNERS)
def test_wrapper_matches_generated_declaration_shard(owner: str) -> None:
    """The runtime seed progression equals the expanded ledger shard."""
    shard = _load_shard(owner)
    wrapper = get_extension("feature", owner)

    assert shard.extension() is wrapper
    assert tuple(shard.CAPABILITY_REFERENCES) == tuple(wrapper.references)
    assert owner_references("feature", owner) == tuple(wrapper.references)


@pytest.mark.parametrize("owner", _OWNERS)
def test_manifest_exposes_the_declared_feature_metadata(owner: str) -> None:
    manifest = extension_manifest("feature", owner)
    metadata = get_feature_definition(owner)

    assert manifest["schema"] == "fx1.extension-module/v1"
    assert manifest["kind"] == "feature"
    assert manifest["feature"] == owner
    assert manifest["feature_family"] == metadata.family
    assert manifest["lookback"] == metadata.lookback
    assert manifest["point_in_time_safe"] is metadata.point_in_time_safe
    assert manifest["synthetic_only"] is (metadata.family == "synthetic_oracle")
    assert manifest["market_evidence"] is False
    assert manifest["operations"] == ["metadata", "build"]


@pytest.mark.parametrize("owner", _OWNERS)
def test_seed_ids_round_trip_to_their_owner(owner: str) -> None:
    references = owner_references("feature", owner)

    for seed_id in (references[0], references[-1]):
        record = resolve_seed_id(int(seed_id))
        assert record["kind"] == "feature"
        assert record["owner"] == owner
        assert record["feature"] == owner


def test_every_registered_feature_owner_lists_its_wrapper() -> None:
    listed = {entry["owner"] for entry in list_extensions("feature")}
    assert set(_OWNERS) <= listed


def test_unknown_feature_fails_closed() -> None:
    with pytest.raises(KeyError):
        get_extension("feature", "not_a_registered_feature")
