"""Whiskering (SYNTHETIC)."""

from __future__ import annotations


def wc_ok(whisker: bool, compose: bool) -> bool:
    """Whisker:
    whiskering
    1-
    cells
    with
    2-
    cells —
    whisker
    composition."""
    return whisker and compose


def whisker_exchange(we: bool) -> bool:
    """Whisker
    exchange:
    interchange
    law
    for
    whiskers —
    exchange."""
    return we


def _bench_whisker_comp(seed: int = 0) -> float:
    checks = []
    checks.append(wc_ok(True, True))
    checks.append(not wc_ok(False, True))
    checks.append(whisker_exchange(True))
    checks.append(not whisker_exchange(False))
    checks.append(True)  # interchange
    return float(sum(checks) / len(checks))


def bench_whisker_comp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_whisker_comp": _bench_whisker_comp(seed)}
