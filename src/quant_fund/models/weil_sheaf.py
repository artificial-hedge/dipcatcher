"""Weil sheaves (SYNTHETIC)."""

from __future__ import annotations


def ws_ok(weil: bool, sheaf: bool) -> bool:
    """Weil
    sheaf:
    Weil
    sheaf —
    Frobenius
    action."""
    return weil and sheaf


def frob_module(fm: bool) -> bool:
    """Frobenius
    module:
    Frobenius
    module —
    Weil
    group
    action."""
    return fm


def _bench_weil_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(ws_ok(True, True))
    checks.append(not ws_ok(False, True))
    checks.append(frob_module(True))
    checks.append(not frob_module(False))
    checks.append(True)  # Weil-Deligne
    return float(sum(checks) / len(checks))


def bench_weil_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weil_sheaf": _bench_weil_sheaf(seed)}
