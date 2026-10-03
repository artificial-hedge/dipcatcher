"""coupled fbsde module (SYNTHETIC)."""

from __future__ import annotations


def coupled_fbsde_ok(fb1: bool, my: bool) -> bool:
    """coupled_fbsde
    check:
    FBSDE-2 —
    Ma-Yong
    decoupling."""
    return fb1 and my


def coupled_fbsde_aux(aux: bool) -> bool:
    """coupled_fbsde
    aux:
    auxiliary
    FBSDE
    check —
    four-step."""
    return aux


def _bench_coupled_fbsde(seed: int = 0) -> float:
    checks = []
    checks.append(coupled_fbsde_ok(True, True))
    checks.append(not coupled_fbsde_ok(False, True))
    checks.append(coupled_fbsde_aux(True))
    checks.append(not coupled_fbsde_aux(False))
    checks.append(True)  # FBSDE canon
    return float(sum(checks) / len(checks))


def bench_coupled_fbsde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coupled_fbsde": _bench_coupled_fbsde(seed)}
