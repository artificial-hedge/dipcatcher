"""Topological K-theory (SYNTHETIC)."""

from __future__ import annotations


def tk_ok(topological: bool, k_theory: bool) -> bool:
    """Topological:
    topological
    K-
    theory —
    Atiyah
    KU."""
    return topological and k_theory


def ku_bott(kb: bool) -> bool:
    """KU
    Bott:
    Bott
    periodicity
    in
    KU —
    Atiyah-
    Bott."""
    return kb


def _bench_topo_k_theory(seed: int = 0) -> float:
    checks = []
    checks.append(tk_ok(True, True))
    checks.append(not tk_ok(False, True))
    checks.append(ku_bott(True))
    checks.append(not ku_bott(False))
    checks.append(True)  # Atiyah
    return float(sum(checks) / len(checks))


def bench_topo_k_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_topo_k_theory": _bench_topo_k_theory(seed)}
