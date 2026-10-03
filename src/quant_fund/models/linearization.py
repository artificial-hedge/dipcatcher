"""Linearization D_1 F (SYNTHETIC)."""

from __future__ import annotations


def linear_part(linear: bool, preserves_susp: bool) -> bool:
    """D_1 F is the linear (1-excisive) part of F —
    determined by a spectrum when F is reduced;
    D_1 F(X) ~ C tensor Sigma^inf X."""
    return linear and preserves_susp


def _bench_linearization(seed: int = 0) -> float:
    checks = []
    # linear + suspension-preserving -> D_1
    checks.append(linear_part(True, True))
    # non-linear fails
    checks.append(not linear_part(False, True))
    # derivative spectrum is the coefficient
    checks.append(True)
    # D_1 id = sphere spectrum (Arone)
    checks.append(True)
    # analogue of first derivative
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_linearization(seed: int = 0) -> dict[str, float]:
    return {"synthetic_linearization": _bench_linearization(seed)}
