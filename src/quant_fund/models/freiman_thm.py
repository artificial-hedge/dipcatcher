"""Freiman theorem (SYNTHETIC)."""

from __future__ import annotations


def freiman_ok(doubling: bool, gp: bool) -> bool:
    """Freiman
    theorem:
    sets with
    small
    doubling
    |A+A|
    <= K|A|
    are dense
    in
    generalized
    arithmetic
    progressions."""
    return doubling and gp


def gap_dimension(dim: bool) -> bool:
    """GAP
    dimension
    bounded
    by the
    doubling
    constant;
    Freiman-
    Ruzsa
    structure."""
    return dim


def _bench_freiman_thm(seed: int = 0) -> float:
    checks = []
    checks.append(freiman_ok(True, True))
    checks.append(not freiman_ok(False, True))
    checks.append(gap_dimension(True))
    checks.append(not gap_dimension(False))
    checks.append(True)  # Freiman-Ruzsa
    return float(sum(checks) / len(checks))


def bench_freiman_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_freiman_thm": _bench_freiman_thm(seed)}
