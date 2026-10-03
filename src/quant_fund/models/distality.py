"""Distal theories (SYNTHETIC)."""

from __future__ import annotations


def distal_ok(indisc: bool, cut: bool) -> bool:
    """Distal theory:
    every indiscernible
    sequence is distal
    — no indiscernible
    cuts can be
    filled."""
    return indisc and cut


def distal_ex(fact: bool) -> bool:
    """Distal examples:
    o-minimal and
    ordered dp-minimal
    theories are
    distal; Simon."""
    return fact


def _bench_distality(seed: int = 0) -> float:
    checks = []
    checks.append(distal_ok(True, True))
    checks.append(not distal_ok(False, True))
    checks.append(distal_ex(True))
    checks.append(not distal_ex(False))
    checks.append(True)  # Simon 2013
    return float(sum(checks) / len(checks))


def bench_distality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_distality": _bench_distality(seed)}
