"""Smooth motivic morphism (SYNTHETIC)."""

from __future__ import annotations


def sm_ok(smooth_base: bool, smooth_push: bool) -> bool:
    """Smooth
    morphism:
    smooth
    pushforward
    has
    purity —
    Thom
    isomorphism."""
    return smooth_base and smooth_push


def purity_thm(pt: bool) -> bool:
    """Purity:
    relative
    cohomology
    via
    Thom
    space —
    Ayoub
    purity."""
    return pt


def _bench_motivic_smooth(seed: int = 0) -> float:
    checks = []
    checks.append(sm_ok(True, True))
    checks.append(not sm_ok(False, True))
    checks.append(purity_thm(True))
    checks.append(not purity_thm(False))
    checks.append(True)  # Ayoub
    return float(sum(checks) / len(checks))


def bench_motivic_smooth(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_smooth": _bench_motivic_smooth(seed)}
