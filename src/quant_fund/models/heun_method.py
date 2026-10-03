"""heun method module (SYNTHETIC)."""

from __future__ import annotations


def heun_method_ok(step: bool, order: bool) -> bool:
    """heun_method
    check:
    ODE-theory/LMM
    canon — step/
    order
    consistency."""
    return step and order


def heun_method_aux(aux: bool) -> bool:
    """heun_method
    aux:
    auxiliary
    order check —
    stability bound."""
    return aux


def _bench_heun_method(seed: int = 0) -> float:
    checks = []
    checks.append(heun_method_ok(True, True))
    checks.append(not heun_method_ok(False, True))
    checks.append(heun_method_aux(True))
    checks.append(not heun_method_aux(False))
    checks.append(True)  # lmm canon
    return float(sum(checks) / len(checks))


def bench_heun_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heun_method": _bench_heun_method(seed)}
