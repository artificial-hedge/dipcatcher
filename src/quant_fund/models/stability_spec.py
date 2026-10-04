"""Stability spectrum: counting types over finite sets (SYNTHETIC)."""

from __future__ import annotations

from fractions import Fraction


def dlo_type_count(params: list[Fraction]) -> int:
    """Complete 1-types over a finite parameter set in DLO: the points
    themselves plus the cuts between/around them: 2n+1 types."""
    return 2 * len(params) + 1


def acf_type_count(params: list[Fraction]) -> int:
    """In ACF, 1-types over a finite set are: realized (algebraic) types
    for each element plus one generic (transcendental) type: n + 1."""
    return len(params) + 1


def is_stable_growth(type_fn, sizes: list[int]) -> bool:
    """A theory is stable-ish when type count grows linearly in |A|; all
    our candidates do, so check linear bound n -> 2n+1 dominated."""
    return all(type_fn([Fraction(i) for i in range(k)]) <= 2 * k + 1 for k in sizes)


def _bench_stability_spec(seed: int = 0) -> float:
    checks = []
    params = [Fraction(0), Fraction(1), Fraction(2)]
    # over 3 parameters DLO has 7 types: {x=0},{x=1},{x=2},{x<0},{0<x<1},
    # {1<x<2},{x>2}
    checks.append(dlo_type_count(params) == 7)
    checks.append(dlo_type_count([Fraction(0)]) == 3)
    # ACF: n params -> n algebraic types + 1 generic
    checks.append(acf_type_count(params) == 4)
    checks.append(acf_type_count([Fraction(0)]) == 2)
    # both are "stable" by the growth criterion (linear, not exponential)
    checks.append(is_stable_growth(dlo_type_count, [1, 2, 5, 10]))
    checks.append(is_stable_growth(acf_type_count, [1, 2, 5, 10]))
    return float(sum(checks) / len(checks))


def bench_stability_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stability_spec": _bench_stability_spec(seed)}
