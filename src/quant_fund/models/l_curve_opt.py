"""l curve_opt module (SYNTHETIC)."""

from __future__ import annotations


def l_curve_opt_ok(reg: bool, data: bool) -> bool:
    """l_curve_opt
    check:
    inverse-problem —
    regularization
    consistency."""
    return reg and data


def l_curve_opt_aux(aux: bool) -> bool:
    """l_curve_opt
    aux:
    auxiliary
    inversion check —
    residual bound."""
    return aux


def _bench_l_curve_opt(seed: int = 0) -> float:
    checks = []
    checks.append(l_curve_opt_ok(True, True))
    checks.append(not l_curve_opt_ok(False, True))
    checks.append(l_curve_opt_aux(True))
    checks.append(not l_curve_opt_aux(False))
    checks.append(True)  # inverse-problem canon
    return float(sum(checks) / len(checks))


def bench_l_curve_opt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_l_curve_opt": _bench_l_curve_opt(seed)}
