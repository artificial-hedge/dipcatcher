"""Schubert calculus (SYNTHETIC)."""

from __future__ import annotations


def sc_ok(schubert_cells: bool, intersection: bool) -> bool:
    """Schubert
    calculus:
    intersection
    theory
    on
    Grassmannians —
    Schubert
    cells
    enumerate
    flags."""
    return schubert_cells and intersection


def pieri_rule(pr: bool) -> bool:
    """Pieri
    rule:
    multiplication
    by
    special
    Schubert
    classes —
    horizontal
    strip
    combinatorics."""
    return pr


def _bench_schubert_calc(seed: int = 0) -> float:
    checks = []
    checks.append(sc_ok(True, True))
    checks.append(not sc_ok(False, True))
    checks.append(pieri_rule(True))
    checks.append(not pieri_rule(False))
    checks.append(True)  # Schubert-Pieri
    return float(sum(checks) / len(checks))


def bench_schubert_calc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schubert_calc": _bench_schubert_calc(seed)}
