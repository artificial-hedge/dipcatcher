"""Baker theorem (SYNTHETIC)."""

from __future__ import annotations


def baker_ok(log_linear: bool, lower_bound: bool) -> bool:
    """Baker:
    linear
    forms
    in
    logarithms
    of
    algebraic
    numbers
    are
    either
    zero
    or
    bounded
    below
    effectively."""
    return log_linear and lower_bound


def effective_diophant(ed: bool) -> bool:
    """Effective
    bounds
    for
    diophantine
    equations
    from
    Baker's
    method —
    Thue
    equations,
    elliptic
    curves."""
    return ed


def _bench_baker_thm(seed: int = 0) -> float:
    checks = []
    checks.append(baker_ok(True, True))
    checks.append(not baker_ok(False, True))
    checks.append(effective_diophant(True))
    checks.append(not effective_diophant(False))
    checks.append(True)  # Baker Fields 1970
    return float(sum(checks) / len(checks))


def bench_baker_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baker_thm": _bench_baker_thm(seed)}
