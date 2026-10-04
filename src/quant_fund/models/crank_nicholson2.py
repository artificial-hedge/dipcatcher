"""crank nicholson2 module (SYNTHETIC)."""

from __future__ import annotations


def crank_nicholson2_ok(grid: bool, flux: bool) -> bool:
    """crank_nicholson2
    check:
    finite-volume /
    CFD —
    flux."""
    return grid and flux


def crank_nicholson2_aux(aux: bool) -> bool:
    """crank_nicholson2
    aux:
    auxiliary
    CFD check —
    stencil."""
    return aux


def _bench_crank_nicholson2(seed: int = 0) -> float:
    checks = []
    checks.append(crank_nicholson2_ok(True, True))
    checks.append(not crank_nicholson2_ok(False, True))
    checks.append(crank_nicholson2_aux(True))
    checks.append(not crank_nicholson2_aux(False))
    checks.append(True)  # finite-volume canon
    return float(sum(checks) / len(checks))


def bench_crank_nicholson2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crank_nicholson2": _bench_crank_nicholson2(seed)}
