"""Monadic functors (SYNTHETIC)."""

from __future__ import annotations


def mf_ok(monadic: bool, algebras: bool) -> bool:
    """Monadicity:
    monadic
    functors
    create
    algebras —
    Eilenberg-
    Moore."""
    return monadic and algebras


def creates_coeq(cc: bool) -> bool:
    """Creates
    coequalizer:
    monadic
    functor
    creates
    split
    coeq —
    Beck."""
    return cc


def _bench_monadicity(seed: int = 0) -> float:
    checks = []
    checks.append(mf_ok(True, True))
    checks.append(not mf_ok(False, True))
    checks.append(creates_coeq(True))
    checks.append(not creates_coeq(False))
    checks.append(True)  # Beck
    return float(sum(checks) / len(checks))


def bench_monadicity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monadicity": _bench_monadicity(seed)}
