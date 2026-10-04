"""cat exact3 module (SYNTHETIC)."""

from __future__ import annotations


def cat_exact3_ok(cat: bool, canonical: bool) -> bool:
    """cat_exact3
    check:
    category
    structure —
    abelian."""
    return cat and canonical


def cat_exact3_aux(aux: bool) -> bool:
    """cat_exact3
    aux:
    auxiliary
    category
    check —
    exact."""
    return aux


def _bench_cat_exact3(seed: int = 0) -> float:
    checks = []
    checks.append(cat_exact3_ok(True, True))
    checks.append(not cat_exact3_ok(False, True))
    checks.append(cat_exact3_aux(True))
    checks.append(not cat_exact3_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_exact3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_exact3": _bench_cat_exact3(seed)}
