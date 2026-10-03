"""Tate triples (SYNTHETIC)."""

from __future__ import annotations


def tt_ok(tate: bool, triple: bool) -> bool:
    """Tate
    triple:
    Tate
    triple
    structures —
    graded
    ring."""
    return tate and triple


def tate_graded(tg: bool) -> bool:
    """Tate
    graded:
    Tate
    triple
    grading —
    weight."""
    return tg


def _bench_tate_triple(seed: int = 0) -> float:
    checks = []
    checks.append(tt_ok(True, True))
    checks.append(not tt_ok(False, True))
    checks.append(tate_graded(True))
    checks.append(not tate_graded(False))
    checks.append(True)  # Tate triples
    return float(sum(checks) / len(checks))


def bench_tate_triple(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tate_triple": _bench_tate_triple(seed)}
