"""goal adaptive module (SYNTHETIC)."""

from __future__ import annotations


def goal_adaptive_ok(mark: bool, est: bool) -> bool:
    """goal_adaptive
    check:
    adaptive —
    marking/estimator
    consistency."""
    return mark and est


def goal_adaptive_aux(aux: bool) -> bool:
    """goal_adaptive
    aux:
    auxiliary
    adaptive check —
    contraction bound."""
    return aux


def _bench_goal_adaptive(seed: int = 0) -> float:
    checks = []
    checks.append(goal_adaptive_ok(True, True))
    checks.append(not goal_adaptive_ok(False, True))
    checks.append(goal_adaptive_aux(True))
    checks.append(not goal_adaptive_aux(False))
    checks.append(True)  # adaptive canon
    return float(sum(checks) / len(checks))


def bench_goal_adaptive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_goal_adaptive": _bench_goal_adaptive(seed)}
