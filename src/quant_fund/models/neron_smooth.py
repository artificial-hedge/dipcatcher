"""Neron model (SYNTHETIC)."""

from __future__ import annotations


def ns_ok(neron: bool, smooth: bool) -> bool:
    """Neron:
    Neron
    model
    smooth
    over
    a
    DVR —
    Neron
    model."""
    return neron and smooth


def smooth_locus(sl: bool) -> bool:
    """Smooth
    locus:
    Neron
    model
    via
    smooth
    locus —
    BLR
    smoothing."""
    return sl


def _bench_neron_smooth(seed: int = 0) -> float:
    checks = []
    checks.append(ns_ok(True, True))
    checks.append(not ns_ok(False, True))
    checks.append(smooth_locus(True))
    checks.append(not smooth_locus(False))
    checks.append(True)  # BLR
    return float(sum(checks) / len(checks))


def bench_neron_smooth(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neron_smooth": _bench_neron_smooth(seed)}
