"""The generated declaration ledger reaches the public extension registry.

``scripts/generated_capability_declarations/features/<owner>.py`` is the
expanded, one-call-per-seed form of the compact seed layout in
:mod:`fx1.capabilities`. Each shard names its runtime wrapper
(``fx1.extensions.features.<owner>.MODULE``) and registers the same seed ids.

Until the compact seed runtime existed, importing a shard -- and therefore
every wrapper in :mod:`fx1.extensions` -- raised ``ModuleNotFoundError``, so
the whole path was unreachable. These tests drive the ledger shards through
their owner and back out of the public ``fx1.extensions`` API.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from fx1.extensions import extension_manifest, get_extension

_OWNERS = (
    "adv",
    "adv_ratio_20_60",
    "amihud",
    "amihud_60",
    "beta_60",
    "cs_z_planted_signal",
    "cs_z_ret_1",
    "dollar_volume",
    "downside_vol_20",
    "high_52w_prox",
    "idio_mom_20",
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


def test_unknown_feature_fails_closed() -> None:
    with pytest.raises(KeyError):
        get_extension("feature", "not_a_registered_feature")
