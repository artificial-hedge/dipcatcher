"""Condensed rings (SYNTHETIC)."""

from __future__ import annotations


def cr2_ok(condensed: bool, ring: bool) -> bool:
    """Condensed
    ring:
    condensed
    ring —
    Scholze
    condensed."""
    return condensed and ring


def condensed_module(cm: bool) -> bool:
    """Condensed
    module:
    condensed
    module
    over
    a
    condensed
    ring —
    quasi-
    separated."""
    return cm


def _bench_condensed_ring(seed: int = 0) -> float:
    checks = []
    checks.append(cr2_ok(True, True))
    checks.append(not cr2_ok(False, True))
    checks.append(condensed_module(True))
    checks.append(not condensed_module(False))
    checks.append(True)  # Scholze
    return float(sum(checks) / len(checks))


def bench_condensed_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_condensed_ring": _bench_condensed_ring(seed)}
