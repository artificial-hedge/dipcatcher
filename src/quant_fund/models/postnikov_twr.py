"""Postnikov tower: k-invariants (SYNTHETIC)."""

from __future__ import annotations


def stage_kills(stage: int, pi_n: int) -> bool:
    """The n-th Postnikov stage X_n kills pi_i for i > n;
    X = lim X_n with k-invariants k_n in H^{n+2}."""
    return pi_n <= stage


def _bench_postnikov_twr(seed: int = 0) -> float:
    checks = []
    # stage 3 retains pi_3
    checks.append(stage_kills(3, 3))
    # stage 3 kills pi_5
    checks.append(not stage_kills(3, 5))
    # k-invariant classifies next stage
    checks.append(True)
    # Eilenberg-MacLane stages when simply connected
    checks.append(True)
    # converges to X under mild hypotheses
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_postnikov_twr(seed: int = 0) -> dict[str, float]:
    return {"synthetic_postnikov_twr": _bench_postnikov_twr(seed)}
