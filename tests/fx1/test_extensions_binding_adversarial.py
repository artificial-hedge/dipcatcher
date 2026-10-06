"""SYNTHETIC adversarial tests for the fx-1 extension binding layer.

``get_extension`` is the only load path for first-party extensions: an owner
id resolves through a fixed whitelist into one permitted module path, and the
imported module's ``MODULE`` record must re-declare its own identity. These
probes attack the naming and binding contracts directly — hostile owner
spellings, kind mixups, identity-mismatched modules, and ``records``
pagination bounds — instead of relying on the ledger happy paths.

All fixtures are synthetic; results are correctness checks, never market
evidence.
"""

from __future__ import annotations

import pytest

from fx1.extensions.contracts import ExtensionModule
from fx1.extensions.naming import module_basename, module_directory, module_path
from fx1.extensions.registry import get_extension

# ------------------------------- naming --------------------------------------


@pytest.mark.parametrize(
    "owner",
    [
        "",
        "UPPER",
        "9leading",
        "has space",
        "dots.name",
        "slash/name",
        "..",
        "../etc",
    ],
)
def test_module_basename_spelling_gate(owner: str) -> None:
    with pytest.raises(ValueError, match="unsafe extension owner"):
        module_basename(owner)


def test_module_basename_maps_hyphen_to_underscore() -> None:
    assert module_basename("kronos-forecast") == "kronos_forecast"


def test_module_path_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown extension kind"):
        module_path("widget", "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown extension kind"):
        module_directory("widget")  # type: ignore[arg-type]


def test_module_path_is_fixed_by_kind_and_owner() -> None:
    assert module_path("plugin", "world-bank") == "fx1.extensions.plugins.world_bank"
    assert module_path("skill", "doctor") == "fx1.extensions.skills.doctor"
    assert module_path("feature", "ret-1") == "fx1.extensions.features.ret_1"


# ------------------------- ExtensionModule binding ---------------------------


def _feature_module(**overrides: object) -> ExtensionModule:
    kwargs: dict[str, object] = {
        "kind": "feature",
        "owner": "ret_1",
        "references": [1],
        "module": "fx1.extensions.features.ret_1",
    }
    kwargs.update(overrides)
    return ExtensionModule(**kwargs)  # type: ignore[arg-type]


def test_extension_module_rejects_wrong_module_path() -> None:
    with pytest.raises(ValueError, match="must live at"):
        _feature_module(module="fx1.extensions.features.other")


def test_extension_module_rejects_kind_owner_mismatch() -> None:
    # Owner "ret_1" under kind "plugin" resolves a plugins path — refusing
    # binds the tuple to exactly one module spelling.
    with pytest.raises(ValueError, match="must live at"):
        _feature_module(kind="plugin")


def test_extension_module_rejects_empty_references() -> None:
    with pytest.raises(ValueError, match="no capability records"):
        _feature_module(references=[])


def test_underscore_owners_are_legal() -> None:
    # Underscores are inside the safe alphabet — owners are underscore-native.
    assert module_basename("ret_1") == "ret_1"


def test_extension_module_manifest_is_read_only_shape() -> None:
    manifest = _feature_module(references=[7, 8]).manifest()
    assert manifest == {
        "schema": "fx1.extension-module/v1",
        "kind": "feature",
        "owner": "ret_1",
        "module": "fx1.extensions.features.ret_1",
        "card_count": 2,
        "market_evidence": False,
    }


@pytest.mark.parametrize("bad_limit", [0, 101, -1])
def test_records_rejects_out_of_range_limit(bad_limit: int) -> None:
    module = _feature_module(references=[1, 2, 3])
    with pytest.raises(ValueError, match="between 1 and 100"):
        list(module.records(limit=bad_limit))


def test_records_rejects_negative_offset() -> None:
    module = _feature_module(references=[1])
    with pytest.raises(ValueError, match="non-negative"):
        list(module.records(offset=-1))


# ------------------------------ get_extension --------------------------------


def test_get_extension_unknown_owner_fails_closed() -> None:
    with pytest.raises(KeyError, match="unknown feature extension"):
        get_extension("feature", "not-a-real-owner")


def test_get_extension_rejects_owner_of_wrong_kind() -> None:
    # "doctor" is a real skill owner but not a plugin — kind confusion must
    # fail closed rather than import an arbitrary module.
    with pytest.raises(KeyError, match="unknown plugin extension"):
        get_extension("plugin", "doctor")


def test_get_extension_hyphen_owner_stays_fail_closed() -> None:
    # Hyphen spellings are basename-safe but owners are underscore-native —
    # the whitelist lookup refuses before any import attempt.
    with pytest.raises(KeyError, match="unknown feature extension"):
        get_extension("feature", "ret-1")


def test_get_extension_real_feature_binds_identity() -> None:
    module = get_extension("feature", "ret_1")
    assert module.kind == "feature"
    assert module.owner == "ret_1"
    assert module.module == "fx1.extensions.features.ret_1"
