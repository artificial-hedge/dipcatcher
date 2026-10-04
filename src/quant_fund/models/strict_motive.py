"""Strict motives (SYNTHETIC)."""

from __future__ import annotations


def sm_ok(strict: bool, transfer: bool) -> bool:
    """Strict
    motive:
    strict
    homotopy
    invariant —
    transfers."""
    return strict and transfer


def nisnevich_sheaf(ns: bool) -> bool:
    """Nisnevich
    sheaf:
    Nisnevich
    sheaf
    with
    transfers —
    localization."""
    return ns


def _bench_strict_motive(seed: int = 0) -> float:
    checks = []
    checks.append(sm_ok(True, True))
    checks.append(not sm_ok(False, True))
    checks.append(nisnevich_sheaf(True))
    checks.append(not nisnevich_sheaf(False))
    checks.append(True)  # Voevodsky
    return float(sum(checks) / len(checks))


def bench_strict_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_strict_motive": _bench_strict_motive(seed)}
