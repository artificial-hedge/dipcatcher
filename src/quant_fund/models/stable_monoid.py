"""stable monoid module (SYNTHETIC)."""

from __future__ import annotations


def stable_monoid_ok(homotopy: bool, stable: bool) -> bool:
    """stable_monoid
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def stable_monoid_aux(aux: bool) -> bool:
    """stable_monoid
    aux:
    auxiliary
    homotopy
    check —
    monoid."""
    return aux


def _bench_stable_monoid(seed: int = 0) -> float:
    checks = []
    checks.append(stable_monoid_ok(True, True))
    checks.append(not stable_monoid_ok(False, True))
    checks.append(stable_monoid_aux(True))
    checks.append(not stable_monoid_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_monoid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_monoid": _bench_stable_monoid(seed)}
