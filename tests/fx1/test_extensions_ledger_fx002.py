"""The fx-002 declaration shards reach the public extension registry.

The feature shards owned by this slice are independently generated
registrations: each names its runtime wrapper
``fx1.extensions.features.<owner>.MODULE`` and calls ``_register(seed_id)`` for
every seed id recorded in :mod:`fx1.capabilities`. Before the compact seed
runtime existed, importing a shard -- and therefore every wrapper in
:mod:`fx1.extensions` -- raised ``ModuleNotFoundError``, so the whole ledger
path was unreachable. These tests drive each shard through its runtime owner
and back out of the public ``fx1.extensions`` API.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from fx1.capabilities import owner_references, resolve_seed_id
from fx1.extensions import extension_manifest, get_extension

_OWNERS = (
    "idio_vol_60",
    "kurt_20",
    "log_price",
    "log_ret_1",
    "max_ret_20",
    "min_ret_20",
    "mom_126",
    "mom_12_1",
    "mom_20",
    "mom_252",
    "mom_5",
    "mom_60",
    "mom_skip_5_20",
    "planted_signal",
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
    assert path.is_file(), f"missing declaration shard {path}"
    spec = importlib.util.spec_from_file_location(f"_capability_ledger_fx002_{owner}", path)
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


@pytest.mark.parametrize("owner", _OWNERS)
def test_shard_seed_ids_resolve_back_to_the_owner(owner: str) -> None:
    shard = _load_shard(owner)
    references = tuple(shard.CAPABILITY_REFERENCES)
    assert references == owner_references("feature", owner)
    assert references

    for seed_id in (references[0], references[len(references) // 2], references[-1]):
        record = resolve_seed_id(seed_id)
        assert record["kind"] == "feature"
        assert record["owner"] == owner
        assert record["feature"] == owner


@pytest.mark.parametrize("owner", _OWNERS)
def test_runtime_wrapper_verify_covers_every_registered_seed(owner: str) -> None:
    wrapper = get_extension("feature", owner)
    wrapper.verify()  # raises if any registered seed resolves to a foreign owner
