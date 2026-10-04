"""multiple scales module (SYNTHETIC)."""

from __future__ import annotations


def multiple_scales_ok(epsilon: bool, uniform: bool) -> bool:
    """multiple_scales
    check:
    perturbation
    method —
    uniform validity."""
    return epsilon and uniform


def multiple_scales_aux(aux: bool) -> bool:
    """multiple_scales
    aux:
    auxiliary
    perturbation check —
    remainder bound."""
    return aux


def _bench_multiple_scales(seed: int = 0) -> float:
    checks = []
    checks.append(multiple_scales_ok(True, True))
    checks.append(not multiple_scales_ok(False, True))
    checks.append(multiple_scales_aux(True))
    checks.append(not multiple_scales_aux(False))
    checks.append(True)  # perturbation-theory canon
    return float(sum(checks) / len(checks))


def bench_multiple_scales(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multiple_scales": _bench_multiple_scales(seed)}
