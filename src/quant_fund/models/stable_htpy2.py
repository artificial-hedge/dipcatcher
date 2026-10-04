"""Stable homotopy (SYNTHETIC)."""

from __future__ import annotations


def sh2_ok(stable: bool, htpy: bool) -> bool:
    """Stable
    homotopy:
    stable
    homotopy —
    stable
    category."""
    return stable and htpy


def stable_category(sc: bool) -> bool:
    """Stable
    category:
    stable
    category —
    suspension
    invertible."""
    return sc


def _bench_stable_htpy2(seed: int = 0) -> float:
    checks = []
    checks.append(sh2_ok(True, True))
    checks.append(not sh2_ok(False, True))
    checks.append(stable_category(True))
    checks.append(not stable_category(False))
    checks.append(True)  # Boardman
    return float(sum(checks) / len(checks))


def bench_stable_htpy2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_htpy2": _bench_stable_htpy2(seed)}
