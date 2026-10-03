"""fx-012: the generated feature wrappers reach the public extension registry.

Each module in ``src/fx1/extensions/features/`` is a generated wrapper bound by
``fx1.extensions.registry.get_extension``. The wrapper imports the compact seed
runtime :mod:`fx1.capabilities` at module scope; before that module existed the
import raised ``ModuleNotFoundError`` and every wrapper in this slice was
unreachable through the public ``fx1.extensions`` API. These tests drive the
public surface for exactly the owners in this slice.

The seed records are generated discovery ids, not market evidence.
"""

from __future__ import annotations

import pytest

from fx1.extensions import extension_manifest, get_extension, list_extensions
from fx1.extensions.feature_catalog import get_feature_definition

_OWNERS = (
    "amihud",
    "amihud_60",
    "beta_60",
    "cs_z_planted_signal",
    "cs_z_ret_1",
    "dollar_volume",
    "downside_vol_20",
    "high_52w_prox",
    "idio_mom_20",
    "idio_vol_60",
    "kurt_20",
    "log_price",
    "log_ret_1",
    "max_ret_20",
)


def test_slice_owners_are_listed_in_the_feature_registry() -> None:
    listed = {entry["owner"] for entry in list_extensions("feature")}
    assert set(_OWNERS) <= listed


@pytest.mark.parametrize("owner", _OWNERS)
def test_feature_wrapper_is_reachable_through_public_api(owner: str) -> None:
    extension = get_extension("feature", owner)

    assert extension.owner == owner
    assert extension.kind == "feature"
    assert extension.module == f"fx1.extensions.features.{owner}"
    assert extension.card_count == len(extension.references) > 0
    assert extension.metadata() == get_feature_definition(owner)


@pytest.mark.parametrize("owner", _OWNERS)
def test_feature_manifest_is_honest_and_addressable(owner: str) -> None:
    manifest = extension_manifest("feature", owner)

    assert manifest["schema"] == "fx1.extension-module/v1"
    assert manifest["kind"] == "feature"
    assert manifest["feature"] == owner
    assert manifest["module"] == f"fx1.extensions.features.{owner}"
    assert manifest["market_evidence"] is False
    assert manifest["operations"] == ["metadata", "build"]


@pytest.mark.parametrize("owner", _OWNERS)
def test_feature_records_are_owned_by_this_wrapper(owner: str) -> None:
    extension = get_extension("feature", owner)
    records = list(extension.records(limit=3))

    assert records
    for record in records:
        assert record["kind"] == "feature"
        assert record["feature"] == owner
        assert record["owner"] == owner


def test_unregistered_feature_still_fails_closed() -> None:
    with pytest.raises(KeyError):
        get_extension("feature", "not_a_registered_feature")
