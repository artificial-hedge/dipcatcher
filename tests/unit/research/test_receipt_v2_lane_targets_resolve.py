"""Lane-target resolve gate — no receipt contract may point at a missing module.

``receipt_v2`` dispatches receipt ``kind``/``schema`` to a consistency checker
by string module path, resolved lazily through :func:`_lane_checker`. That
indirection is deliberate (a lane may land after its contract is declared), but
it also means a typo or an unlanded module silently degrades a deep check into
``<label>_lane_missing`` — a receipt then verifies against nothing at all.

This gate enumerates every declared lane target and asserts it either resolves
via ``importlib`` or is listed in ``PENDING_LANE_CONSISTENCY`` with a non-empty
reason. The pending list is shrink-only: an entry whose module has since landed
fails here, so the exemptions cannot rot.
"""

from __future__ import annotations

import importlib

import pytest

from quant_fund.research.receipt_v2 import (
    _LANE_CONSISTENCY,
    _LANE_CONSISTENCY_OUTER_KIND,
    PENDING_LANE_CONSISTENCY,
    PENDING_V1_CONTRACT,
    _kind_consistency_errors,
)


def _resolves(module_path: str, func_name: str) -> bool:
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return False
    return callable(getattr(module, func_name, None))


def _all_declared_targets() -> list[tuple[str, str, str, str]]:
    """(registry, kind, module, func) for every declared lane target."""
    out: list[tuple[str, str, str, str]] = []
    for kind, dotted in _LANE_CONSISTENCY.items():
        module, _, func = dotted.rpartition(".")
        out.append(("_LANE_CONSISTENCY", kind, module, func))
    for kind, (module, func, _label) in _LANE_CONSISTENCY_OUTER_KIND.items():
        out.append(("_LANE_CONSISTENCY_OUTER_KIND", kind, module, func))
    return out


def test_declared_targets_found_something() -> None:
    """Guard the guard: an empty enumeration would make every check vacuous."""
    targets = _all_declared_targets()
    assert len(targets) >= 10, f"only {len(targets)} lane targets enumerated — scan broken"


def test_every_target_resolves_or_is_documented_pending() -> None:
    for registry, kind, module, func in _all_declared_targets():
        if _resolves(module, func):
            continue
        assert kind in PENDING_LANE_CONSISTENCY, (
            f"{registry}[{kind!r}] -> {module}.{func} does not resolve and the kind "
            f"is not in PENDING_LANE_CONSISTENCY — land the module, fix the path, "
            f"or document the pending lane with a reason"
        )


def test_pending_lanes_have_reasons_and_stay_pending() -> None:
    for kind, (target, reason) in PENDING_LANE_CONSISTENCY.items():
        module, func, _label = target
        assert reason.strip(), f"pending lane {kind!r} lacks a documented reason"
        # Shrink-only: once the module lands the exemption is stale and the
        # lane must move into the resolving set on its own merits.
        assert not _resolves(module, func), (
            f"pending lane {kind!r} now resolves at {module}.{func} — "
            f"drop it from PENDING_LANE_CONSISTENCY"
        )
        assert kind in _LANE_CONSISTENCY_OUTER_KIND, (
            f"pending lane {kind!r} is not dispatched — add it to "
            f"_LANE_CONSISTENCY_OUTER_KIND so it still fails closed"
        )


def test_pending_v1_contract_is_documented() -> None:
    module, func, label = PENDING_V1_CONTRACT
    assert label.strip(), "PENDING_V1_CONTRACT needs a non-empty label"
    if not _resolves(module, func):
        assert "hstep_bench" in PENDING_LANE_CONSISTENCY, (
            f"{module}.{func} does not resolve and no pending lane documents it"
        )


@pytest.mark.parametrize("kind", sorted(PENDING_LANE_CONSISTENCY))
def test_pending_lane_fails_closed(kind: str) -> None:
    """A receipt claiming a pending kind is never silently accepted."""
    _module, _func, label = PENDING_LANE_CONSISTENCY[kind][0]
    errors = _kind_consistency_errors({"kind": kind})
    assert errors == [f"{label}_lane_missing"], (
        f"kind {kind!r} must fail closed with {label}_lane_missing, got {errors}"
    )


def test_landed_lanes_never_report_lane_missing() -> None:
    """Converse guard: a resolving lane must not degrade to lane_missing."""
    for kind, (module, func, label) in _LANE_CONSISTENCY_OUTER_KIND.items():
        if kind in PENDING_LANE_CONSISTENCY:
            continue
        assert _resolves(module, func), f"{kind} -> {module}.{func} vanished"
        errors = _kind_consistency_errors({"kind": kind, "payload": {}})
        assert f"{label}_lane_missing" not in errors, (
            f"lane {kind!r} resolved but still reported {label}_lane_missing"
        )


def test_inner_claim_lanes_resolve() -> None:
    """The inner-claim map holds only landed lanes — no pending entries there."""
    for kind, dotted in _LANE_CONSISTENCY.items():
        module, _, func = dotted.rpartition(".")
        assert _resolves(module, func), f"_LANE_CONSISTENCY[{kind!r}] -> {dotted} does not resolve"
        assert kind not in PENDING_LANE_CONSISTENCY, (
            f"{kind!r} is pending yet sits in the inner-claim map"
        )


def test_one_lane_contract_per_kind() -> None:
    """The dispatcher breaks after the first match, so a kind must not be in both."""
    overlap = set(_LANE_CONSISTENCY) & set(_LANE_CONSISTENCY_OUTER_KIND)
    assert not overlap, f"kinds declared in both lane maps: {sorted(overlap)}"
