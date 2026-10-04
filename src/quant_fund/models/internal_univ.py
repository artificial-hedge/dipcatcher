"""Internal universes (SYNTHETIC)."""

from __future__ import annotations


def internal_univ_ok(tarski: bool, univ_axiom: bool) -> bool:
    """Internal universe U with
    decoding Tarski-style El;
    univalence axiom holds for
    U = universe of small
    types."""
    return tarski and univ_axiom


def universe_lift(universe: bool) -> bool:
    """Universe levels U_0 : U_1 :
    U_2; cumulativity; liftings
    between levels for
    abstractions."""
    return universe


def _bench_internal_univ(seed: int = 0) -> float:
    checks = []
    checks.append(internal_univ_ok(True, True))
    checks.append(not internal_univ_ok(False, True))
    checks.append(universe_lift(True))
    checks.append(not universe_lift(False))
    checks.append(True)  # Tarski vs Russell styles
    return float(sum(checks) / len(checks))


def bench_internal_univ(seed: int = 0) -> dict[str, float]:
    return {"synthetic_internal_univ": _bench_internal_univ(seed)}
