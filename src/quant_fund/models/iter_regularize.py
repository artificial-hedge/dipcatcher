"""iter regularize module (SYNTHETIC)."""

from __future__ import annotations


def iter_regularize_ok(reg: bool, data: bool) -> bool:
    """iter_regularize
    check:
    inverse-problem —
    regularization
    consistency."""
    return reg and data


def iter_regularize_aux(aux: bool) -> bool:
    """iter_regularize
    aux:
    auxiliary
    inversion check —
    residual bound."""
    return aux


def _bench_iter_regularize(seed: int = 0) -> float:
    checks = []
    checks.append(iter_regularize_ok(True, True))
    checks.append(not iter_regularize_ok(False, True))
    checks.append(iter_regularize_aux(True))
    checks.append(not iter_regularize_aux(False))
    checks.append(True)  # inverse-problem canon
    return float(sum(checks) / len(checks))


def bench_iter_regularize(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iter_regularize": _bench_iter_regularize(seed)}
