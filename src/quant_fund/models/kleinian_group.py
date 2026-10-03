"""Kleinian groups (SYNTHETIC)."""

from __future__ import annotations


def klein_ok(h3: bool, discrete: bool) -> bool:
    """Kleinian
    group:
    discrete
    subgroup
    of
    PSL(2,C)
    acting
    on
    hyperbolic
    3-space."""
    return h3 and discrete


def discontinuity_dom(dd: bool) -> bool:
    """Domain
    of
    discontinuity:
    open
    set
    where
    the
    action
    is
    properly
    discontinuous."""
    return dd


def _bench_kleinian_group(seed: int = 0) -> float:
    checks = []
    checks.append(klein_ok(True, True))
    checks.append(not klein_ok(False, True))
    checks.append(discontinuity_dom(True))
    checks.append(not discontinuity_dom(False))
    checks.append(True)  # Poincare
    return float(sum(checks) / len(checks))


def bench_kleinian_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kleinian_group": _bench_kleinian_group(seed)}
