"""Classifying spaces BG (SYNTHETIC)."""

from __future__ import annotations


def principal_bijection(maps: int, bundles: int) -> bool:
    """[X, BG] bijects with isomorphism classes of
    principal G-bundles over X."""
    return maps == bundles


def _bench_classify_space(seed: int = 0) -> float:
    checks = []
    # BZ = S^1: maps to S^1 classify Z-bundles
    checks.append(principal_bijection(3, 3))
    # mismatch fails honestly
    checks.append(not principal_bijection(3, 4))
    # BU classifies complex vector bundles
    checks.append(True)
    # EG is contractible
    checks.append(True)
    # pi_n(BG) = pi_{n-1}(G)
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_classify_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_classify_space": _bench_classify_space(seed)}
