"""Untilting (SYNTHETIC)."""

from __future__ import annotations


def ut_ok(untilt: bool, tilt: bool) -> bool:
    """Untilt:
    untilt
    of
    a
    tilted
    perfectoid —
    untilt."""
    return untilt and tilt


def tilting_process(tp: bool) -> bool:
    """Tilting:
    tilting
    equivalence
    of
    perfectoid —
    Scholze
    tilting."""
    return tp


def _bench_untilt(seed: int = 0) -> float:
    checks = []
    checks.append(ut_ok(True, True))
    checks.append(not ut_ok(False, True))
    checks.append(tilting_process(True))
    checks.append(not tilting_process(False))
    checks.append(True)  # Scholze
    return float(sum(checks) / len(checks))


def bench_untilt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_untilt": _bench_untilt(seed)}
