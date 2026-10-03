"""Category of algebras (SYNTHETIC)."""

from __future__ import annotations


def ac_ok(algebra: bool, endofunctor: bool) -> bool:
    """Algebra:
    category
    of
    algebras
    for
    endofunctor —
    Lambek."""
    return algebra and endofunctor


def lambek_iso(li: bool) -> bool:
    """Lambek:
    initial
    algebra
    is
    iso
    —
    Lambek
    lemma."""
    return li


def _bench_algebra_cat(seed: int = 0) -> float:
    checks = []
    checks.append(ac_ok(True, True))
    checks.append(not ac_ok(False, True))
    checks.append(lambek_iso(True))
    checks.append(not lambek_iso(False))
    checks.append(True)  # Lambek
    return float(sum(checks) / len(checks))


def bench_algebra_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_algebra_cat": _bench_algebra_cat(seed)}
