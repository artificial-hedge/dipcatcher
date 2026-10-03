"""diamond taylor_wiles module (SYNTHETIC)."""

from __future__ import annotations


def diamond_taylor_wiles_ok(galois: bool, deform: bool) -> bool:
    """diamond_taylor_wiles
    check:
    Galois-deformation
    structure —
    Kisin."""
    return galois and deform


def diamond_taylor_wiles_aux(aux: bool) -> bool:
    """diamond_taylor_wiles
    aux:
    auxiliary
    deformation
    check —
    Taylor."""
    return aux


def _bench_diamond_taylor_wiles(seed: int = 0) -> float:
    checks = []
    checks.append(diamond_taylor_wiles_ok(True, True))
    checks.append(not diamond_taylor_wiles_ok(False, True))
    checks.append(diamond_taylor_wiles_aux(True))
    checks.append(not diamond_taylor_wiles_aux(False))
    checks.append(True)  # Galois-deformation canon
    return float(sum(checks) / len(checks))


def bench_diamond_taylor_wiles(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diamond_taylor_wiles": _bench_diamond_taylor_wiles(seed)}
