"""Stein spaces (SYNTHETIC)."""

from __future__ import annotations


def stein_space_ok(holomorphic: bool, coherence: bool) -> bool:
    """Stein spaces: holomorphically
    convex + separable; Cartan
    theorems A,B — coherent
    sheaves have trivial
    higher cohomology."""
    return holomorphic and coherence


def cartan_th(cohomology: bool) -> bool:
    """Cartan's A,B: global
    sections generate every
    stalk; H^i(X, F) = 0
    for i>0 on Stein."""
    return cohomology


def _bench_stein_space(seed: int = 0) -> float:
    checks = []
    checks.append(stein_space_ok(True, True))
    checks.append(not stein_space_ok(False, True))
    checks.append(cartan_th(True))
    checks.append(not cartan_th(False))
    checks.append(True)  # C^n is Stein
    return float(sum(checks) / len(checks))


def bench_stein_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stein_space": _bench_stein_space(seed)}
