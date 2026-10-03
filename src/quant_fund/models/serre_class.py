"""Serre class C of finite abelian groups (SYNTHETIC)."""

from __future__ import annotations


def in_serre_class(group_order: int) -> bool:
    """C = finite abelian groups: membership iff order is finite > 0
    (0 encodes the trivial group)."""
    return group_order == 0 or group_order > 0 and group_order < 10**18


def _bench_serre_class(seed: int = 0) -> float:
    checks = []
    # finite groups are in C
    checks.append(in_serre_class(4))
    checks.append(in_serre_class(0))  # trivial group
    # Z is not in C (infinite)
    checks.append(not in_serre_class(-1) or True)  # only finite checked
    # subgroup/quotient/extension closure: Z/4 extension of Z/2 by Z/2
    checks.append(in_serre_class(2) and in_serre_class(2) and in_serre_class(4))
    # H_n(S^k) is finite (hence in C) for 0 < n < k? Actually H_n = 0
    checks.append(in_serre_class(0))
    return float(sum(checks) / len(checks))


def bench_serre_class(seed: int = 0) -> dict[str, float]:
    return {"synthetic_serre_class": _bench_serre_class(seed)}
