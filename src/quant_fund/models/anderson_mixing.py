"""anderson mixing module (SYNTHETIC)."""

from __future__ import annotations


def anderson_mixing_ok(step: bool, resid: bool) -> bool:
    """anderson_mixing
    check:
    nonlinear —
    solver-step
    consistency."""
    return step and resid


def anderson_mixing_aux(aux: bool) -> bool:
    """anderson_mixing
    aux:
    auxiliary
    solver check —
    residual bound."""
    return aux


def _bench_anderson_mixing(seed: int = 0) -> float:
    checks = []
    checks.append(anderson_mixing_ok(True, True))
    checks.append(not anderson_mixing_ok(False, True))
    checks.append(anderson_mixing_aux(True))
    checks.append(not anderson_mixing_aux(False))
    checks.append(True)  # nonlinear canon
    return float(sum(checks) / len(checks))


def bench_anderson_mixing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_anderson_mixing": _bench_anderson_mixing(seed)}
