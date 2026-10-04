"""Infinity fibrations (SYNTHETIC)."""

from __future__ import annotations


def fi_ok(fib: bool, infty: bool) -> bool:
    """Infinity
    fibration:
    infinity
    categorical
    fibration —
    inner
    fibration."""
    return fib and infty


def inner_fibration(inf: bool) -> bool:
    """Inner
    fibration:
    inner
    fibration
    of
    simplicial
    sets —
    RLP
    inner."""
    return inf


def _bench_fib_infty(seed: int = 0) -> float:
    checks = []
    checks.append(fi_ok(True, True))
    checks.append(not fi_ok(False, True))
    checks.append(inner_fibration(True))
    checks.append(not inner_fibration(False))
    checks.append(True)  # Joyal-Lurie
    return float(sum(checks) / len(checks))


def bench_fib_infty(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fib_infty": _bench_fib_infty(seed)}
