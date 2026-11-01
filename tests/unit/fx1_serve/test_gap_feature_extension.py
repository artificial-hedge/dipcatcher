"""Unit tests for the generated feature-extension wrappers.

``fx1.extensions.features.*`` are generated one-line modules that bind a
public feature column to the existing PIT feature registry. The contract is
small but load-bearing: a wrapper that names the wrong owner, or points at
the wrong seed range, silently attaches the wrong feature to a research
column.

The specific module covered here is ``z_vs_ma20``; the assertions are
written against the shared contract so they hold for any wrapper in the
family.
"""

from __future__ import annotations

from fx1.capabilities import owner_references
from fx1.extensions.features import z_vs_ma20 as z_vs_ma20_module


def test_the_wrapper_binds_the_declared_feature_owner() -> None:
    assert z_vs_ma20_module.MODULE.owner == "z_vs_ma20"
    assert z_vs_ma20_module.MODULE.kind == "feature"


def test_the_wrapper_points_at_its_own_module_path() -> None:
    assert z_vs_ma20_module.MODULE.module == "fx1.extensions.features.z_vs_ma20"


def test_the_wrapper_uses_the_registered_seed_range_for_its_owner() -> None:
    """The seed references must be the ones the capability registry declares."""
    assert z_vs_ma20_module.MODULE.references == owner_references("feature", "z_vs_ma20")


def test_the_wrapper_resolves_the_feature_metadata() -> None:
    metadata = z_vs_ma20_module.MODULE.metadata()
    assert metadata.name == "z_vs_ma20"
    assert metadata.lookback == 20
    assert metadata.point_in_time_safe is True


def test_the_seed_references_are_a_non_empty_strictly_increasing_range() -> None:
    references = z_vs_ma20_module.MODULE.references
    assert references
    assert all(b > a for a, b in zip(references, references[1:], strict=False))
