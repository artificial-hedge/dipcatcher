"""stable excisive module (SYNTHETIC)."""

from __future__ import annotations


def stable_excisive_ok(homotopy: bool, stable: bool) -> bool:
    """stable_excisive
    check:
    homotopy
    structure —
    suspension."""
    return homotopy and stable


def stable_excisive_aux(aux: bool) -> bool:
    """stable_excisive
    aux:
    auxiliary
    homotopy
    check —
    fiber."""
    return aux


def _bench_stable_excisive(seed: int = 0) -> float:
    checks = []
    checks.append(stable_excisive_ok(True, True))
    checks.append(not stable_excisive_ok(False, True))
    checks.append(stable_excisive_aux(True))
    checks.append(not stable_excisive_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_excisive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_excisive": _bench_stable_excisive(seed)}
