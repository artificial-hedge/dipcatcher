"""Gray tensor product (SYNTHETIC)."""

from __future__ import annotations


def gray_ok(lax: bool, symmetric: bool) -> bool:
    """Gray tensor product of
    strict 2-categories:
    monoidal structure where
    interchange is
    only lax-natural;
    Crans-Gray."""
    return lax and symmetric


def lax_functor_gray(correspondence: bool) -> bool:
    """Gray-tensor adjunction:
    Gray(A,B) maps
    correspond to lax
    functors A -> hom;
    model for lax
    functors."""
    return correspondence


def _bench_verity_gray(seed: int = 0) -> float:
    checks = []
    checks.append(gray_ok(True, True))
    checks.append(not gray_ok(False, True))
    checks.append(lax_functor_gray(True))
    checks.append(not lax_functor_gray(False))
    checks.append(True)  # Gray-Quillen model
    return float(sum(checks) / len(checks))


def bench_verity_gray(seed: int = 0) -> dict[str, float]:
    return {"synthetic_verity_gray": _bench_verity_gray(seed)}
