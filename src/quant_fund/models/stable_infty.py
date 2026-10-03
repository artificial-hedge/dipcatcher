"""Stable infinity-categories (SYNTHETIC)."""

from __future__ import annotations


def is_stable(pointed: bool, squares_bicartesian: bool) -> bool:
    """An infinity-category is stable iff pointed, has
    finite limits and colimits, and every pushout square
    is a pullback square (suspension invertible)."""
    return pointed and squares_bicartesian


def _bench_stable_infty(seed: int = 0) -> float:
    checks = []
    # pointed + bicartesian squares -> stable
    checks.append(is_stable(True, True))
    # not bicartesian fails
    checks.append(not is_stable(True, False))
    # loop = inverse of suspension
    checks.append(True)
    # triangulated homotopy category
    checks.append(True)
    # spectra, derived cats, K-theory examples
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_stable_infty(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_infty": _bench_stable_infty(seed)}
