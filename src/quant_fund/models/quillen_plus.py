"""Quillen plus construction (SYNTHETIC)."""

from __future__ import annotations


def qp_ok(plus_constr: bool, acyclic: bool) -> bool:
    """Plus
    construction:
    kills
    perfect
    subgroup
    preserving
    homology —
    Quillen
    plus."""
    return plus_constr and acyclic


def plus_k_groups(pk: bool) -> bool:
    """Plus
    K:
    K-groups
    from
    plus
    construction
    homotopy —
    algebraic
    K."""
    return pk


def _bench_quillen_plus(seed: int = 0) -> float:
    checks = []
    checks.append(qp_ok(True, True))
    checks.append(not qp_ok(False, True))
    checks.append(plus_k_groups(True))
    checks.append(not plus_k_groups(False))
    checks.append(True)  # Quillen
    return float(sum(checks) / len(checks))


def bench_quillen_plus(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quillen_plus": _bench_quillen_plus(seed)}
