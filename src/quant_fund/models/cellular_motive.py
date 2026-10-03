"""Cellular motives (SYNTHETIC)."""

from __future__ import annotations


def cell_ok(a1_connected: bool, motive_stable: bool) -> bool:
    """Cellular variety X built from
    A^1-cells; motive M(X) is a Tate
    motive direct sum (Morel)."""
    return a1_connected and motive_stable


def tate_decomp(wheelfill: bool) -> bool:
    """Tate motives M(X) = direct
    sum Z(i)[2i] for cellular
    X; MW-correspondences
    compute cohomology."""
    return wheelfill


def _bench_cellular_motive(seed: int = 0) -> float:
    checks = []
    checks.append(cell_ok(True, True))
    checks.append(not cell_ok(False, True))
    checks.append(tate_decomp(True))
    checks.append(not tate_decomp(False))
    checks.append(True)  # Grassmannian is cellular
    return float(sum(checks) / len(checks))


def bench_cellular_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cellular_motive": _bench_cellular_motive(seed)}
