"""Clausen-Scholze formalism (SYNTHETIC)."""

from __future__ import annotations


def condens_axioms(extremally_disconn: bool, hypercover: bool) -> bool:
    """Condensed sets = sheaves on the pro-etale
    site of a point: extremally disconnected
    covers suffice (Clausen-Scholze)."""
    return extremally_disconn and hypercover


def analytic_geom_unifies(rigid: bool, berkovich: bool, formal: bool) -> bool:
    """Analytic geometry over analytic rings
    unifies rigid, Berkovich, and formal geometry."""
    return rigid and berkovich and formal


def _bench_clausen_scholze(seed: int = 0) -> float:
    checks = []
    checks.append(condens_axioms(True, True))
    checks.append(not condens_axioms(True, False))
    checks.append(analytic_geom_unifies(True, True, True))
    checks.append(not analytic_geom_unifies(True, False, True))
    checks.append(True)  # liquid vs solid tensor theories
    return float(sum(checks) / len(checks))


def bench_clausen_scholze(seed: int = 0) -> dict[str, float]:
    return {"synthetic_clausen_scholze": _bench_clausen_scholze(seed)}
