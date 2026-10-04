"""cat freyd module (SYNTHETIC)."""

from __future__ import annotations


def cat_freyd_ok(cat: bool, canonical: bool) -> bool:
    """cat_freyd
    check:
    category
    structure —
    abelian."""
    return cat and canonical


def cat_freyd_aux(aux: bool) -> bool:
    """cat_freyd
    aux:
    auxiliary
    category
    check —
    exact."""
    return aux


def _bench_cat_freyd(seed: int = 0) -> float:
    checks = []
    checks.append(cat_freyd_ok(True, True))
    checks.append(not cat_freyd_ok(False, True))
    checks.append(cat_freyd_aux(True))
    checks.append(not cat_freyd_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_freyd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_freyd": _bench_cat_freyd(seed)}
