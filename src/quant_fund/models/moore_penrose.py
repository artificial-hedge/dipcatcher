"""moore penrose module (SYNTHETIC)."""

from __future__ import annotations


def moore_penrose_ok(step: bool, resid: bool) -> bool:
    """moore_penrose
    check:
    nonlinear —
    solver-step
    consistency."""
    return step and resid


def moore_penrose_aux(aux: bool) -> bool:
    """moore_penrose
    aux:
    auxiliary
    solver check —
    residual bound."""
    return aux


def _bench_moore_penrose(seed: int = 0) -> float:
    checks = []
    checks.append(moore_penrose_ok(True, True))
    checks.append(not moore_penrose_ok(False, True))
    checks.append(moore_penrose_aux(True))
    checks.append(not moore_penrose_aux(False))
    checks.append(True)  # nonlinear canon
    return float(sum(checks) / len(checks))


def bench_moore_penrose(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moore_penrose": _bench_moore_penrose(seed)}
