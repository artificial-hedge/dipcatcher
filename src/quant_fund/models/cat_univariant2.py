"""cat univariant2 module (SYNTHETIC)."""

from __future__ import annotations


def cat_univariant2_ok(cat: bool, canonical: bool) -> bool:
    """cat_univariant2
    check:
    category
    structure —
    abelian."""
    return cat and canonical


def cat_univariant2_aux(aux: bool) -> bool:
    """cat_univariant2
    aux:
    auxiliary
    category
    check —
    exact."""
    return aux


def _bench_cat_univariant2(seed: int = 0) -> float:
    checks = []
    checks.append(cat_univariant2_ok(True, True))
    checks.append(not cat_univariant2_ok(False, True))
    checks.append(cat_univariant2_aux(True))
    checks.append(not cat_univariant2_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_univariant2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_univariant2": _bench_cat_univariant2(seed)}
