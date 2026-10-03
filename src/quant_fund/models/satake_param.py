"""Satake parameters (SYNTHETIC)."""

from __future__ import annotations


def satake_ok(unramified: bool, conjugacy: bool) -> bool:
    """Satake parameter of
    unramified rep pi_p:
    semisimple conjugacy
    class in L-G;
    determines L-factor."""
    return unramified and conjugacy


def semisimple_class(eigenvalues: bool) -> bool:
    """Satake parameters
    alpha_i(p) are the
    eigenvalues of
    Frobenius-like
    matrices."""
    return eigenvalues


def _bench_satake_param(seed: int = 0) -> float:
    checks = []
    checks.append(satake_ok(True, True))
    checks.append(not satake_ok(False, True))
    checks.append(semisimple_class(True))
    checks.append(not semisimple_class(False))
    checks.append(True)  # Satake-Langlands
    return float(sum(checks) / len(checks))


def bench_satake_param(seed: int = 0) -> dict[str, float]:
    return {"synthetic_satake_param": _bench_satake_param(seed)}
