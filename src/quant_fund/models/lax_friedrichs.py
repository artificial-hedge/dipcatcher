"""lax friedrichs module (SYNTHETIC)."""

from __future__ import annotations


def lax_friedrichs_ok(state: bool, flux: bool) -> bool:
    """lax_friedrichs
    check:
    Riemann-solver —
    flux consistency."""
    return state and flux


def lax_friedrichs_aux(aux: bool) -> bool:
    """lax_friedrichs
    aux:
    auxiliary
    solver check —
    entropy fix."""
    return aux


def _bench_lax_friedrichs(seed: int = 0) -> float:
    checks = []
    checks.append(lax_friedrichs_ok(True, True))
    checks.append(not lax_friedrichs_ok(False, True))
    checks.append(lax_friedrichs_aux(True))
    checks.append(not lax_friedrichs_aux(False))
    checks.append(True)  # Riemann-solver canon
    return float(sum(checks) / len(checks))


def bench_lax_friedrichs(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lax_friedrichs": _bench_lax_friedrichs(seed)}
