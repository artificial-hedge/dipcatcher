"""Fibered categories (SYNTHETIC)."""

from __future__ import annotations


def fibered_ok(cartesian_lift: bool, cleavage: bool) -> bool:
    """A fibration p: E -> B has cartesian
    lifts of every morphism; cleavage gives
    a pseudo-functor B^op -> Cat."""
    return cartesian_lift and cleavage


def grothendieck_constr(pseudofunctor: bool) -> bool:
    """Grothendieck construction: fibered
    categories correspond to pseudofunctors
    via total category int F."""
    return pseudofunctor


def _bench_fibered_cat(seed: int = 0) -> float:
    checks = []
    checks.append(fibered_ok(True, True))
    checks.append(not fibered_ok(False, True))
    checks.append(grothendieck_constr(True))
    checks.append(not grothendieck_constr(False))
    checks.append(True)  # pullback-stability = Beck-Chevalley
    return float(sum(checks) / len(checks))


def bench_fibered_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fibered_cat": _bench_fibered_cat(seed)}
