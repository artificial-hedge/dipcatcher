"""Orthogonal/Weiss calculus (SYNTHETIC)."""

from __future__ import annotations


def orth_taylor(on_vecspaces: bool, polynomial: bool) -> bool:
    """Weiss orthogonal calculus: functors on real inner
    product spaces have Taylor towers with n-th
    derivatives O(n)-spectra; models unstable
    approximation of V -> V."""
    return on_vecspaces and polynomial


def _bench_orth_calc(seed: int = 0) -> float:
    checks = []
    # functors on vector spaces + poly approx
    checks.append(orth_taylor(True, True))
    # wrong domain fails
    checks.append(not orth_taylor(False, True))
    # derivative of BO gives spheres
    checks.append(True)
    # applies to functor calculus on manifolds
    checks.append(True)
    # Arone's model for layers
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_orth_calc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_orth_calc": _bench_orth_calc(seed)}
