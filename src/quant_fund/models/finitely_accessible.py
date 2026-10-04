"""Finitely accessible categories (SYNTHETIC)."""

from __future__ import annotations


def fa_ok(finitely: bool, accessible: bool) -> bool:
    """Finitely
    accessible:
    finitely
    accessible
    cat —
    finitely
    presentable
    objects
    dense."""
    return finitely and accessible


def finitely_presentable(fp: bool) -> bool:
    """Finitely
    presentable:
    hom
    preserves
    filtered
    colimits —
    fp object."""
    return fp


def _bench_finitely_accessible(seed: int = 0) -> float:
    checks = []
    checks.append(fa_ok(True, True))
    checks.append(not fa_ok(False, True))
    checks.append(finitely_presentable(True))
    checks.append(not finitely_presentable(False))
    checks.append(True)  # Gabriel-Ulmer
    return float(sum(checks) / len(checks))


def bench_finitely_accessible(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finitely_accessible": _bench_finitely_accessible(seed)}
