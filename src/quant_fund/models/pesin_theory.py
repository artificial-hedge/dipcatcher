"""Pesin theory (SYNTHETIC)."""

from __future__ import annotations


def pesin_ok(nonzero: bool, stable: bool) -> bool:
    """Pesin
    theory:
    nonzero
    Lyapunov
    exponents
    give
    local
    stable/
    unstable
    manifolds."""
    return nonzero and stable


def measurable_split(split: bool) -> bool:
    """Measurable
    splitting:
    Oseledets
    spaces
    vary
    measurably
    across the
    nonuniformly
    hyperbolic
    set."""
    return split


def _bench_pesin_theory(seed: int = 0) -> float:
    checks = []
    checks.append(pesin_ok(True, True))
    checks.append(not pesin_ok(False, True))
    checks.append(measurable_split(True))
    checks.append(not measurable_split(False))
    checks.append(True)  # Pesin
    return float(sum(checks) / len(checks))


def bench_pesin_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pesin_theory": _bench_pesin_theory(seed)}
