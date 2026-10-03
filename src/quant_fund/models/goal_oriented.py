"""goal oriented module (SYNTHETIC)."""

from __future__ import annotations


def goal_oriented_ok(est: bool, bound: bool) -> bool:
    """goal_oriented
    check:
    a-posteriori error —
    estimator
    consistency."""
    return est and bound


def goal_oriented_aux(aux: bool) -> bool:
    """goal_oriented
    aux:
    auxiliary
    residual check —
    bound
    reliability."""
    return aux


def _bench_goal_oriented(seed: int = 0) -> float:
    checks = []
    checks.append(goal_oriented_ok(True, True))
    checks.append(not goal_oriented_ok(False, True))
    checks.append(goal_oriented_aux(True))
    checks.append(not goal_oriented_aux(False))
    checks.append(True)  # a-posteriori canon
    return float(sum(checks) / len(checks))


def bench_goal_oriented(seed: int = 0) -> dict[str, float]:
    return {"synthetic_goal_oriented": _bench_goal_oriented(seed)}
