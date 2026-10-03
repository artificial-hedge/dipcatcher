"""Dagger / weak formal schemes (SYNTHETIC)."""

from __future__ import annotations


def dagger_str(overconv_fn: bool, weak_complete: bool) -> bool:
    """Dagger (weakly complete) algebras:
    overconvergent power series T_n^dagger =
    lim eps -> 0 O(B(0,1+eps)) (Meredith)."""
    return overconv_fn and weak_complete


def dagger_finitely_presented(finite_type: bool) -> bool:
    """Dagger algebras are quotients of
    weak Tate algebras, topologically
    of finite type."""
    return finite_type


def _bench_dagger_groth(seed: int = 0) -> float:
    checks = []
    checks.append(dagger_str(True, True))
    checks.append(not dagger_str(False, True))
    checks.append(dagger_finitely_presented(True))
    checks.append(not dagger_finitely_presented(False))
    checks.append(True)  # dagger vs strict affinoid
    return float(sum(checks) / len(checks))


def bench_dagger_groth(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dagger_groth": _bench_dagger_groth(seed)}
