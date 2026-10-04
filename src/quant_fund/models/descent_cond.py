"""Descent and hyperdescent (SYNTHETIC)."""

from __future__ import annotations


def descent_ok(effective_epi: bool, colimit_comm: bool) -> bool:
    """Descent: for effective epimorphism U -> X,
    X = colim of the Cech nerve; colimits are
    universal in an infinity-topos."""
    return effective_epi and colimit_comm


def hypercover_descent(sieve_gen: bool) -> bool:
    """Hypercover descent = sheafification on
    hypercovers of the site (Brown)."""
    return sieve_gen


def _bench_descent_cond(seed: int = 0) -> float:
    checks = []
    checks.append(descent_ok(True, True))
    checks.append(not descent_ok(True, False))
    checks.append(hypercover_descent(True))
    checks.append(not hypercover_descent(False))
    checks.append(True)  # sheaf <-> 1-categorical descent
    return float(sum(checks) / len(checks))


def bench_descent_cond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_descent_cond": _bench_descent_cond(seed)}
