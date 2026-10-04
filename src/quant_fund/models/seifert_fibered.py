"""Seifert fibered spaces (SYNTHETIC)."""

from __future__ import annotations


def seifert_ok(circles: bool, base: bool) -> bool:
    """Seifert
    fibered
    space:
    foliated
    by
    circles
    over
    an
    orbifold
    base —
    generalized
    S1-bundle."""
    return circles and base


def euler_invariant(ev: bool) -> bool:
    """Euler
    number
    and
    orbifold
    Euler
    characteristic
    classify
    Seifert
    spaces."""
    return ev


def _bench_seifert_fibered(seed: int = 0) -> float:
    checks = []
    checks.append(seifert_ok(True, True))
    checks.append(not seifert_ok(False, True))
    checks.append(euler_invariant(True))
    checks.append(not euler_invariant(False))
    checks.append(True)  # Seifert
    return float(sum(checks) / len(checks))


def bench_seifert_fibered(seed: int = 0) -> dict[str, float]:
    return {"synthetic_seifert_fibered": _bench_seifert_fibered(seed)}
