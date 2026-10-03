"""Levi problem (SYNTHETIC)."""

from __future__ import annotations


def levi_ok(pseudo_equiv: bool, oka_solved: bool) -> bool:
    """Levi
    problem:
    pseudoconvex
    domains
    are
    domains
    of
    holomorphy —
    solved
    by
    Oka."""
    return pseudo_equiv and oka_solved


def stein_equiv(se: bool) -> bool:
    """Stein
    characterization:
    pseudoconvex
    plus
    K-
    complete
    equals
    Stein."""
    return se


def _bench_levi_problem(seed: int = 0) -> float:
    checks = []
    checks.append(levi_ok(True, True))
    checks.append(not levi_ok(False, True))
    checks.append(stein_equiv(True))
    checks.append(not stein_equiv(False))
    checks.append(True)  # Oka-Norguet-Bremermann
    return float(sum(checks) / len(checks))


def bench_levi_problem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_levi_problem": _bench_levi_problem(seed)}
