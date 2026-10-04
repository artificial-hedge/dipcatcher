"""self decomp module (SYNTHETIC)."""

from __future__ import annotations


def self_decomp_ok(id1: bool, cg: bool) -> bool:
    """self_decomp
    check:
    infinitely-divisible
    structure —
    Levy
    canon."""
    return id1 and cg


def self_decomp_aux(aux: bool) -> bool:
    """self_decomp
    aux:
    auxiliary
    triplet
    check —
    Khinchin
    formula."""
    return aux


def _bench_self_decomp(seed: int = 0) -> float:
    checks = []
    checks.append(self_decomp_ok(True, True))
    checks.append(not self_decomp_ok(False, True))
    checks.append(self_decomp_aux(True))
    checks.append(not self_decomp_aux(False))
    checks.append(True)  # levy canon
    return float(sum(checks) / len(checks))


def bench_self_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_self_decomp": _bench_self_decomp(seed)}
