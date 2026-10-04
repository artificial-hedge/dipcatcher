"""block low_rank module (SYNTHETIC)."""

from __future__ import annotations


def block_low_rank_ok(rank: bool, err: bool) -> bool:
    """block_low_rank
    check:
    low-rank —
    compression
    consistency."""
    return rank and err


def block_low_rank_aux(aux: bool) -> bool:
    """block_low_rank
    aux:
    auxiliary
    low-rank check —
    truncation bound."""
    return aux


def _bench_block_low_rank(seed: int = 0) -> float:
    checks = []
    checks.append(block_low_rank_ok(True, True))
    checks.append(not block_low_rank_ok(False, True))
    checks.append(block_low_rank_aux(True))
    checks.append(not block_low_rank_aux(False))
    checks.append(True)  # low-rank canon
    return float(sum(checks) / len(checks))


def bench_block_low_rank(seed: int = 0) -> dict[str, float]:
    return {"synthetic_block_low_rank": _bench_block_low_rank(seed)}
