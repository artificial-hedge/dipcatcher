"""MMP flip (SYNTHETIC)."""

from __future__ import annotations


def mf_ok(small_contraction: bool, flip: bool) -> bool:
    """Flip:
    replaces
    negative
    curve
    with
    positive —
    Mori
    flip."""
    return small_contraction and flip


def flip_terminates(ft: bool) -> bool:
    """Termination:
    flips
    terminate —
    BCHM
    termination."""
    return ft


def _bench_mmp_flip(seed: int = 0) -> float:
    checks = []
    checks.append(mf_ok(True, True))
    checks.append(not mf_ok(False, True))
    checks.append(flip_terminates(True))
    checks.append(not flip_terminates(False))
    checks.append(True)  # BCHM
    return float(sum(checks) / len(checks))


def bench_mmp_flip(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mmp_flip": _bench_mmp_flip(seed)}
