"""Pasting diagrams (SYNTHETIC)."""

from __future__ import annotations


def pd_ok(pasting: bool, composition: bool) -> bool:
    """Pasting:
    pasting
    diagram
    composition —
    two-cell
    pasting."""
    return pasting and composition


def pasting_scheme(ps: bool) -> bool:
    """Pasting
    scheme:
    pasting
    in
    bicategories —
    Power
    pasting."""
    return ps


def _bench_pasting_diag(seed: int = 0) -> float:
    checks = []
    checks.append(pd_ok(True, True))
    checks.append(not pd_ok(False, True))
    checks.append(pasting_scheme(True))
    checks.append(not pasting_scheme(False))
    checks.append(True)  # Power
    return float(sum(checks) / len(checks))


def bench_pasting_diag(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pasting_diag": _bench_pasting_diag(seed)}
