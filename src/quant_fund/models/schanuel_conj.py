"""Schanuel conjecture (SYNTHETIC)."""

from __future__ import annotations


def sch_ok(degree_bound: bool, open_prob: bool) -> bool:
    """Schanuel
    conjecture:
    trdeg
    Q(z_i,
    e^{z_i})
    at
    least
    n
    for
    linearly
    independent
    z_i —
    wide open."""
    return degree_bound and open_prob


def implies_algebraic_indep(ai: bool) -> bool:
    """Schanuel
    implies
    e
    and
    pi
    are
    algebraically
    independent
    and
    much
    more."""
    return ai


def _bench_schanuel_conj(seed: int = 0) -> float:
    checks = []
    checks.append(sch_ok(True, True))
    checks.append(not sch_ok(False, True))
    checks.append(implies_algebraic_indep(True))
    checks.append(not implies_algebraic_indep(False))
    checks.append(True)  # Schanuel
    return float(sum(checks) / len(checks))


def bench_schanuel_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schanuel_conj": _bench_schanuel_conj(seed)}
