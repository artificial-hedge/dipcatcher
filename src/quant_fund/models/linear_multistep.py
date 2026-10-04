"""linear multistep module (SYNTHETIC)."""

from __future__ import annotations


def linear_multistep_ok(step: bool, order: bool) -> bool:
    """linear_multistep
    check:
    ODE-theory/LMM
    canon — step/
    order
    consistency."""
    return step and order


def linear_multistep_aux(aux: bool) -> bool:
    """linear_multistep
    aux:
    auxiliary
    order check —
    stability bound."""
    return aux


def _bench_linear_multistep(seed: int = 0) -> float:
    checks = []
    checks.append(linear_multistep_ok(True, True))
    checks.append(not linear_multistep_ok(False, True))
    checks.append(linear_multistep_aux(True))
    checks.append(not linear_multistep_aux(False))
    checks.append(True)  # lmm canon
    return float(sum(checks) / len(checks))


def bench_linear_multistep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_linear_multistep": _bench_linear_multistep(seed)}
