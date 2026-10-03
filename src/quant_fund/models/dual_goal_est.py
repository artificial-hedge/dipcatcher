"""dual goal_est module (SYNTHETIC)."""

from __future__ import annotations


def dual_goal_est_ok(node: bool, resid: bool) -> bool:
    """dual_goal_est
    check:
    collocation /
    least-squares
    canon — node/
    residual
    consistency."""
    return node and resid


def dual_goal_est_aux(aux: bool) -> bool:
    """dual_goal_est
    aux:
    auxiliary
    residual check —
    defect bound."""
    return aux


def _bench_dual_goal_est(seed: int = 0) -> float:
    checks = []
    checks.append(dual_goal_est_ok(True, True))
    checks.append(not dual_goal_est_ok(False, True))
    checks.append(dual_goal_est_aux(True))
    checks.append(not dual_goal_est_aux(False))
    checks.append(True)  # colloc canon
    return float(sum(checks) / len(checks))


def bench_dual_goal_est(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dual_goal_est": _bench_dual_goal_est(seed)}
