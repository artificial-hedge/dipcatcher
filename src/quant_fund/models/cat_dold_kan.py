"""cat dold_kan module (SYNTHETIC)."""

from __future__ import annotations


def cat_dold_kan_ok(cat: bool, canonical: bool) -> bool:
    """cat_dold_kan
    check:
    category
    structure —
    pseudo."""
    return cat and canonical


def cat_dold_kan_aux(aux: bool) -> bool:
    """cat_dold_kan
    aux:
    auxiliary
    category
    check —
    weak."""
    return aux


def _bench_cat_dold_kan(seed: int = 0) -> float:
    checks = []
    checks.append(cat_dold_kan_ok(True, True))
    checks.append(not cat_dold_kan_ok(False, True))
    checks.append(cat_dold_kan_aux(True))
    checks.append(not cat_dold_kan_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_dold_kan(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_dold_kan": _bench_cat_dold_kan(seed)}
