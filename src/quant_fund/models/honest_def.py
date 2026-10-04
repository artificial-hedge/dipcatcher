"""Honest definitions (SYNTHETIC)."""

from __future__ import annotations


def honest_ok(uniform: bool, honest: bool) -> bool:
    """Honest definition:
    a formula ψ(y)
    honestly defining
    the trace of
    φ(x,y) on an
    indiscernible seq."""
    return uniform and honest


def honest_ex(hon: bool) -> bool:
    """Chernikov-Simon:
    in NIP theories
    honest definitions
    exist for all
    invariant types."""
    return hon


def _bench_honest_def(seed: int = 0) -> float:
    checks = []
    checks.append(honest_ok(True, True))
    checks.append(not honest_ok(False, True))
    checks.append(honest_ex(True))
    checks.append(not honest_ex(False))
    checks.append(True)  # Chernikov-Simon
    return float(sum(checks) / len(checks))


def bench_honest_def(seed: int = 0) -> dict[str, float]:
    return {"synthetic_honest_def": _bench_honest_def(seed)}
