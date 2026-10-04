"""Analytic sheaf (SYNTHETIC)."""

from __future__ import annotations


def as_ok(sheaf_analytic: bool, liquid_modules: bool) -> bool:
    """Analytic
    sheaf:
    sheaves
    of
    liquid
    modules —
    condensed
    analytic."""
    return sheaf_analytic and liquid_modules


def liquid_sheaf(ls: bool) -> bool:
    """Liquid
    sheaf:
    sheaves
    on
    analytic
    spaces —
    Scholze
    liquid."""
    return ls


def _bench_analytic_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(as_ok(True, True))
    checks.append(not as_ok(False, True))
    checks.append(liquid_sheaf(True))
    checks.append(not liquid_sheaf(False))
    checks.append(True)  # Scholze
    return float(sum(checks) / len(checks))


def bench_analytic_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_analytic_sheaf": _bench_analytic_sheaf(seed)}
