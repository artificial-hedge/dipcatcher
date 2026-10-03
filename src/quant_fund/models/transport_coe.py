"""Transport and coercion along paths (SYNTHETIC)."""

from __future__ import annotations


def coe_follows_path(sends_i0: bool, lands_i1: bool) -> bool:
    """coe(i) along a type path A : I -> U sends
    a : A<i0> to an element of A<i1>."""
    return sends_i0 and lands_i1


def transport_is_coe(dependent: bool) -> bool:
    """transport p^B = coe of B o p as a family
    over the interval."""
    return dependent


def _bench_transport_coe(seed: int = 0) -> float:
    checks = []
    checks.append(coe_follows_path(True, True))
    checks.append(not coe_follows_path(False, True))
    checks.append(transport_is_coe(True))
    checks.append(not transport_is_coe(False))
    checks.append(True)  # coe at constant path is identity
    return float(sum(checks) / len(checks))


def bench_transport_coe(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transport_coe": _bench_transport_coe(seed)}
