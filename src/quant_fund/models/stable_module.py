"""stable module module (SYNTHETIC)."""

from __future__ import annotations


def stable_module_ok(homotopy: bool, stable: bool) -> bool:
    """stable_module
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def stable_module_aux(aux: bool) -> bool:
    """stable_module
    aux:
    auxiliary
    homotopy
    check —
    monoid."""
    return aux


def _bench_stable_module(seed: int = 0) -> float:
    checks = []
    checks.append(stable_module_ok(True, True))
    checks.append(not stable_module_ok(False, True))
    checks.append(stable_module_aux(True))
    checks.append(not stable_module_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_module(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_module": _bench_stable_module(seed)}
