"""Adversarial probes for fx1.extensions — SYNTHETIC, deterministic, no network."""

from __future__ import annotations

import pytest

from fx1.extensions import naming, registry
from fx1.extensions.contracts import (
    FeatureExtension,
    PluginExtension,
    SkillExtension,
)
from fx1.extensions.feature_catalog import (
    FEATURE_DEFINITIONS,
    get_feature_definition,
    list_feature_definitions,
)


def test_module_basename_refuses_unsafe_owners() -> None:
    for bad in ("", "9abc", "UPPER", "a b", "a/b", "a.b", "a;b", "a\\b", "..", "a--b/../x"):
        with pytest.raises(ValueError):
            naming.module_basename(bad)


def test_module_basename_maps_dash_to_underscore() -> None:
    assert naming.module_basename("build-features") == "build_features"
    assert naming.module_basename("verify-research") == "verify_research"


def test_module_path_refuses_unknown_kind() -> None:
    with pytest.raises(ValueError):
        naming.module_path("lambda", "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        naming.module_directory("lambda")  # type: ignore[arg-type]


def test_module_path_format() -> None:
    assert naming.module_path("skill", "build-features") == "fx1.extensions.skills.build_features"
    assert naming.module_path("plugin", "yahoo_finance") == "fx1.extensions.plugins.yahoo_finance"
    assert naming.module_path("feature", "ret_1") == "fx1.extensions.features.ret_1"


def test_list_extensions_unique_module_paths() -> None:
    listing = registry.list_extensions()
    paths = [entry["module"] for entry in listing]
    assert len(paths) == len(set(paths))
    assert all(entry["market_evidence"] is False for entry in listing)
    assert all(entry["schema"] == "fx1.extension-module/v1" for entry in listing)


def test_list_extensions_kind_filter() -> None:
    skills = registry.list_extensions("skill")
    assert skills and all(e["kind"] == "skill" for e in skills)
    assert {e["owner"] for e in skills} >= {"doctor", "verify-research"}


def test_list_extensions_collision_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two owners mapping to one module path must refuse the whole listing."""
    monkeypatch.setattr(registry, "_owners", lambda kind: ("a-b", "a_b") if kind == "skill" else ())
    with pytest.raises(ValueError, match="collision"):
        registry.list_extensions("skill")


def test_get_extension_refuses_unknown_owner() -> None:
    with pytest.raises(KeyError):
        registry.get_extension("skill", "no-such-owner")
    with pytest.raises(KeyError):
        registry.get_extension("feature", "no_such_feature")


def test_get_extension_refuses_unknown_kind() -> None:
    with pytest.raises(ValueError):
        registry.get_extension("lambda", "x")  # type: ignore[arg-type]


def test_extension_module_identity_binding() -> None:
    """A module at the right path but claiming a different owner/kind is refused."""
    with pytest.raises(ValueError):
        SkillExtension(
            kind="plugin", owner="doctor", references=[1], module="fx1.extensions.plugins.doctor"
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SkillExtension(
            kind="skill", owner="doctor", references=[], module="fx1.extensions.skills.doctor"
        )


def test_extension_module_wrong_path_refused() -> None:
    with pytest.raises(ValueError):
        PluginExtension(
            kind="plugin",
            owner="yahoo_finance",
            references=[1],
            module="fx1.extensions.plugins.other_name",
        )


def test_records_bounds_enforced() -> None:
    ext = FeatureExtension(
        kind="feature", owner="ret_1", references=[1, 2, 3], module="fx1.extensions.features.ret_1"
    )
    with pytest.raises(ValueError):
        list(ext.records(offset=-1))
    with pytest.raises(ValueError):
        list(ext.records(limit=0))
    with pytest.raises(ValueError):
        list(ext.records(limit=101))


def test_feature_catalog_unique_names_and_pit_flags() -> None:
    names = [f.name for f in list_feature_definitions()]
    assert len(names) == len(set(names))
    by_name = {f.name: f for f in FEATURE_DEFINITIONS}
    assert by_name["planted_signal"].synthetic_only is True
    assert by_name["cs_z_planted_signal"].synthetic_only is True
    assert all(f.lookback >= 0 for f in FEATURE_DEFINITIONS)


def test_feature_catalog_fail_closed_lookup() -> None:
    with pytest.raises(KeyError):
        get_feature_definition("sharpe_ratio")
    with pytest.raises(KeyError):
        get_feature_definition("pnl")


def test_extension_manifest_refuses_unknown_kind() -> None:
    with pytest.raises(ValueError):
        registry.extension_manifest("lambda", "x")  # type: ignore[arg-type]
