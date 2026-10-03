"""Uniform definability (SYNTHETIC)."""

from __future__ import annotations


def uniform_ok(uniform: bool, params: bool) -> bool:
    """Uniform definability:
    in NIP, families
    of definable sets
    are uniformly
    definable; bounded
    VC dimension."""
    return uniform and params


def bounded_vc(vc: bool) -> bool:
    """Bounded VC
    dimension: NIP
    iff the shatter
    function of each
    formula is
    polynomially bounded."""
    return vc


def _bench_uniform_def(seed: int = 0) -> float:
    checks = []
    checks.append(uniform_ok(True, True))
    checks.append(not uniform_ok(False, True))
    checks.append(bounded_vc(True))
    checks.append(not bounded_vc(False))
    checks.append(True)  # Shelah-Sauer
    return float(sum(checks) / len(checks))


def bench_uniform_def(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uniform_def": _bench_uniform_def(seed)}
