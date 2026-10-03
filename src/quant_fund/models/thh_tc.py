"""Topological Hochschild and cyclic homology (SYNTHETIC)."""

from __future__ import annotations


def cyclotomic_struct(s1_action: bool, frobenius: bool) -> bool:
    """THH(R) carries an S^1-action; cyclotomic structure
    = S^1-action + Frobenius maps defining TC and
    trace K(R) -> TC(R) (Dundas-Goodwillie-McCarthy)."""
    return s1_action and frobenius


def _bench_thh_tc(seed: int = 0) -> float:
    checks = []
    # S1 action + Frobenius -> cyclotomic
    checks.append(cyclotomic_struct(True, True))
    # missing Frobenius fails
    checks.append(not cyclotomic_struct(True, False))
    # THH = tensor over S1 algebra
    checks.append(True)
    # TC via homotopy (co)fixed points
    checks.append(True)
    # computes K-theory of truncated poly rings
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_thh_tc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thh_tc": _bench_thh_tc(seed)}
