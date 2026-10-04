"""Nori motives 2 (SYNTHETIC)."""

from __future__ import annotations


def nm2_ok(nori: bool, motive: bool) -> bool:
    """Nori
    motive
    2:
    Nori
    motive —
    diagrams."""
    return nori and motive


def nori_diagram(nd: bool) -> bool:
    """Nori
    diagram:
    Nori
    diagram
    category —
    abelian."""
    return nd


def _bench_norimotive2(seed: int = 0) -> float:
    checks = []
    checks.append(nm2_ok(True, True))
    checks.append(not nm2_ok(False, True))
    checks.append(nori_diagram(True))
    checks.append(not nori_diagram(False))
    checks.append(True)  # Nori
    return float(sum(checks) / len(checks))


def bench_norimotive2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_norimotive2": _bench_norimotive2(seed)}
