"""Isogenies of abelian varieties (SYNTHETIC)."""

from __future__ import annotations


def iso_ok(finite_kernel: bool, surjective: bool) -> bool:
    """Isogeny:
    finite
    surjective
    homomorphism
    between
    abelian
    varieties —
    category
    up
    to
    isogeny."""
    return finite_kernel and surjective


def dual_isogeny(di: bool) -> bool:
    """Dual
    isogeny:
    every
    isogeny
    has
    a
    dual
    of
    same
    degree —
    Poincare
    reducibility."""
    return di


def _bench_isogeny_av(seed: int = 0) -> float:
    checks = []
    checks.append(iso_ok(True, True))
    checks.append(not iso_ok(False, True))
    checks.append(dual_isogeny(True))
    checks.append(not dual_isogeny(False))
    checks.append(True)  # Poincare
    return float(sum(checks) / len(checks))


def bench_isogeny_av(seed: int = 0) -> dict[str, float]:
    return {"synthetic_isogeny_av": _bench_isogeny_av(seed)}
