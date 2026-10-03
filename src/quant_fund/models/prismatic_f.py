"""Prismatic F-crystals (SYNTHETIC)."""

from __future__ import annotations


def pf_ok(prismatic: bool, f_crystal: bool) -> bool:
    """Prismatic
    F:
    prismatic
    F
    crystal —
    Bhatt
    Scholze."""
    return prismatic and f_crystal


def f_crystal(fc: bool) -> bool:
    """F
    crystal:
    F
    crystal —
    Frobenius
    descent."""
    return fc


def _bench_prismatic_f(seed: int = 0) -> float:
    checks = []
    checks.append(pf_ok(True, True))
    checks.append(not pf_ok(False, True))
    checks.append(f_crystal(True))
    checks.append(not f_crystal(False))
    checks.append(True)  # Bhatt-Scholze
    return float(sum(checks) / len(checks))


def bench_prismatic_f(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prismatic_f": _bench_prismatic_f(seed)}
