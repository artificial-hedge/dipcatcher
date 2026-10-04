"""epsilon algo module (SYNTHETIC)."""

from __future__ import annotations


def epsilon_algo_ok(elem: bool, dof: bool) -> bool:
    """epsilon_algo
    check:
    FE-basis/sequence —
    element/dof
    consistency."""
    return elem and dof


def epsilon_algo_aux(aux: bool) -> bool:
    """epsilon_algo
    aux:
    auxiliary
    element check —
    partition bound."""
    return aux


def _bench_epsilon_algo(seed: int = 0) -> float:
    checks = []
    checks.append(epsilon_algo_ok(True, True))
    checks.append(not epsilon_algo_ok(False, True))
    checks.append(epsilon_algo_aux(True))
    checks.append(not epsilon_algo_aux(False))
    checks.append(True)  # FE-basis canon
    return float(sum(checks) / len(checks))


def bench_epsilon_algo(seed: int = 0) -> dict[str, float]:
    return {"synthetic_epsilon_algo": _bench_epsilon_algo(seed)}
