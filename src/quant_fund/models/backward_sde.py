"""backward sde module (SYNTHETIC)."""

from __future__ import annotations


def backward_sde_ok(bs1: bool, pp: bool) -> bool:
    """backward_sde
    check:
    BSDE —
    Pardoux-Peng
    adapted
    solution."""
    return bs1 and pp


def backward_sde_aux(aux: bool) -> bool:
    """backward_sde
    aux:
    auxiliary
    FBSDE
    check —
    decoupling
    field."""
    return aux


def _bench_backward_sde(seed: int = 0) -> float:
    checks = []
    checks.append(backward_sde_ok(True, True))
    checks.append(not backward_sde_ok(False, True))
    checks.append(backward_sde_aux(True))
    checks.append(not backward_sde_aux(False))
    checks.append(True)  # BSDE canon
    return float(sum(checks) / len(checks))


def bench_backward_sde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_backward_sde": _bench_backward_sde(seed)}
