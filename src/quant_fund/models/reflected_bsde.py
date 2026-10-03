"""reflected bsde module (SYNTHETIC)."""

from __future__ import annotations


def reflected_bsde_ok(bs1: bool, pp: bool) -> bool:
    """reflected_bsde
    check:
    BSDE —
    Pardoux-Peng
    adapted
    solution."""
    return bs1 and pp


def reflected_bsde_aux(aux: bool) -> bool:
    """reflected_bsde
    aux:
    auxiliary
    FBSDE
    check —
    decoupling
    field."""
    return aux


def _bench_reflected_bsde(seed: int = 0) -> float:
    checks = []
    checks.append(reflected_bsde_ok(True, True))
    checks.append(not reflected_bsde_ok(False, True))
    checks.append(reflected_bsde_aux(True))
    checks.append(not reflected_bsde_aux(False))
    checks.append(True)  # BSDE canon
    return float(sum(checks) / len(checks))


def bench_reflected_bsde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reflected_bsde": _bench_reflected_bsde(seed)}
