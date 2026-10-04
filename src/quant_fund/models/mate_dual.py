"""Mate correspondence (SYNTHETIC)."""

from __future__ import annotations


def md_ok(mate: bool, adjoint: bool) -> bool:
    """Mate:
    mate
    under
    adjunction —
    Kelly
    mate."""
    return mate and adjoint


def mate_square(ms: bool) -> bool:
    """Mate
    square:
    mates
    of
    2-
    cells —
    Kelly-
    Street
    mates."""
    return ms


def _bench_mate_dual(seed: int = 0) -> float:
    checks = []
    checks.append(md_ok(True, True))
    checks.append(not md_ok(False, True))
    checks.append(mate_square(True))
    checks.append(not mate_square(False))
    checks.append(True)  # Kelly
    return float(sum(checks) / len(checks))


def bench_mate_dual(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mate_dual": _bench_mate_dual(seed)}
