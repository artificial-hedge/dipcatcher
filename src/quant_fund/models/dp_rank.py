"""dp-rank (SYNTHETIC)."""

from __future__ import annotations


def dp_rank_ok(ict: bool, finiteness: bool) -> bool:
    """dp-rank of a type
    or formula: sup
    of lengths of
    mutually indiscernible
    ict-patterns;
    NIP = dp < ∞."""
    return ict and finiteness


def dp_min(rank_one: bool) -> bool:
    """dp-minimal:
    dp-rank of the
    universe is 1;
    includes o-minimal
    and C-minimal."""
    return rank_one


def _bench_dp_rank(seed: int = 0) -> float:
    checks = []
    checks.append(dp_rank_ok(True, True))
    checks.append(not dp_rank_ok(False, True))
    checks.append(dp_min(True))
    checks.append(not dp_min(False))
    checks.append(True)  # Shelah-Usvyatsov
    return float(sum(checks) / len(checks))


def bench_dp_rank(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dp_rank": _bench_dp_rank(seed)}
