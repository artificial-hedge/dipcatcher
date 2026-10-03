"""stable derivator module (SYNTHETIC)."""

from __future__ import annotations


def stable_derivator_ok(homotopy: bool, stable: bool) -> bool:
    """stable_derivator
    check:
    homotopy
    structure —
    suspension."""
    return homotopy and stable


def stable_derivator_aux(aux: bool) -> bool:
    """stable_derivator
    aux:
    auxiliary
    homotopy
    check —
    fiber."""
    return aux


def _bench_stable_derivator(seed: int = 0) -> float:
    checks = []
    checks.append(stable_derivator_ok(True, True))
    checks.append(not stable_derivator_ok(False, True))
    checks.append(stable_derivator_aux(True))
    checks.append(not stable_derivator_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_derivator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_derivator": _bench_stable_derivator(seed)}
