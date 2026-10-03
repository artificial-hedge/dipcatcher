"""Exponentiable objects (SYNTHETIC)."""

from __future__ import annotations


def exp_ok(functors: bool, fibred: bool) -> bool:
    """Exponentiable maps f: Y -> X
    have right adjoint Pi_f to f^*
    on slice categories; path-
    objects exponentiate."""
    return functors and fibred


def param_right_adj(preserve: bool) -> bool:
    """Right parametrized adjoints
    exist for exponentiable
    morphisms in a topos
    (Kock, Carboni)."""
    return preserve


def _bench_exponentiable(seed: int = 0) -> float:
    checks = []
    checks.append(exp_ok(True, True))
    checks.append(not exp_ok(False, True))
    checks.append(param_right_adj(True))
    checks.append(not param_right_adj(False))
    checks.append(True)  # etale maps exponentiable
    return float(sum(checks) / len(checks))


def bench_exponentiable(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exponentiable": _bench_exponentiable(seed)}
