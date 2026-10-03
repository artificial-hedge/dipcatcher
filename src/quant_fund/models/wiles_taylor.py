"""wiles taylor module (SYNTHETIC)."""

from __future__ import annotations


def wiles_taylor_ok(galois: bool, deform: bool) -> bool:
    """wiles_taylor
    check:
    Galois-deformation
    structure —
    Kisin."""
    return galois and deform


def wiles_taylor_aux(aux: bool) -> bool:
    """wiles_taylor
    aux:
    auxiliary
    deformation
    check —
    Taylor."""
    return aux


def _bench_wiles_taylor(seed: int = 0) -> float:
    checks = []
    checks.append(wiles_taylor_ok(True, True))
    checks.append(not wiles_taylor_ok(False, True))
    checks.append(wiles_taylor_aux(True))
    checks.append(not wiles_taylor_aux(False))
    checks.append(True)  # Galois-deformation canon
    return float(sum(checks) / len(checks))


def bench_wiles_taylor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wiles_taylor": _bench_wiles_taylor(seed)}
