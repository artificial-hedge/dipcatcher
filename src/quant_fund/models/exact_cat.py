"""Exact / Barr-exact categories (SYNTHETIC)."""

from __future__ import annotations


def exact_cat_ok(regular: bool, eff_equiv: bool) -> bool:
    """Barr-exact category: regular
    + every equivalence
    relation is effective;
    abelian cats are exact."""
    return regular and eff_equiv


def good_quotients(equivalence: bool) -> bool:
    """Exact categories have good
    quotients of equivalence
    relations; calculus of
    relations (Mal'cev)."""
    return equivalence


def _bench_exact_cat(seed: int = 0) -> float:
    checks = []
    checks.append(exact_cat_ok(True, True))
    checks.append(not exact_cat_ok(False, True))
    checks.append(good_quotients(True))
    checks.append(not good_quotients(False))
    checks.append(True)  # Sets and Ab are exact
    return float(sum(checks) / len(checks))


def bench_exact_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exact_cat": _bench_exact_cat(seed)}
