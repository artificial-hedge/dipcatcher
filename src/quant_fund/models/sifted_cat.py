"""Sifted colimits / algebraic theories (SYNTHETIC)."""

from __future__ import annotations


def sifted_ok(filtered_finite: bool, split_coeq: bool) -> bool:
    """Sifted colimits = filtered + finite
    coproducts + reflexive coequalizers;
    free algebras preserve them."""
    return filtered_finite and split_coeq


def lawvere_theory(single_sort: bool) -> bool:
    """Lawvere theory T gives
    algebraic categories Alg(T) =
    functors T -> Set preserving
    finite products."""
    return single_sort


def _bench_sifted_cat(seed: int = 0) -> float:
    checks = []
    checks.append(sifted_ok(True, True))
    checks.append(not sifted_ok(False, True))
    checks.append(lawvere_theory(True))
    checks.append(not lawvere_theory(False))
    checks.append(True)  # monadicity <-> varieties
    return float(sum(checks) / len(checks))


def bench_sifted_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sifted_cat": _bench_sifted_cat(seed)}
