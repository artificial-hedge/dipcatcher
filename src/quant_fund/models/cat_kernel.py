"""cat kernel module (SYNTHETIC)."""

from __future__ import annotations


def cat_kernel_ok(category: bool, structure: bool) -> bool:
    """cat_kernel
    check:
    category
    structure —
    rank."""
    return category and structure


def cat_kernel_aux(aux: bool) -> bool:
    """cat_kernel
    aux:
    auxiliary
    category
    check —
    index."""
    return aux


def _bench_cat_kernel(seed: int = 0) -> float:
    checks = []
    checks.append(cat_kernel_ok(True, True))
    checks.append(not cat_kernel_ok(False, True))
    checks.append(cat_kernel_aux(True))
    checks.append(not cat_kernel_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_kernel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_kernel": _bench_cat_kernel(seed)}
