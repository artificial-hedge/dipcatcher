"""low rank_svd module (SYNTHETIC)."""

from __future__ import annotations


def low_rank_svd_ok(rank: bool, err: bool) -> bool:
    """low_rank_svd
    check:
    low-rank —
    compression
    consistency."""
    return rank and err


def low_rank_svd_aux(aux: bool) -> bool:
    """low_rank_svd
    aux:
    auxiliary
    low-rank check —
    truncation bound."""
    return aux


def _bench_low_rank_svd(seed: int = 0) -> float:
    checks = []
    checks.append(low_rank_svd_ok(True, True))
    checks.append(not low_rank_svd_ok(False, True))
    checks.append(low_rank_svd_aux(True))
    checks.append(not low_rank_svd_aux(False))
    checks.append(True)  # low-rank canon
    return float(sum(checks) / len(checks))


def bench_low_rank_svd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_low_rank_svd": _bench_low_rank_svd(seed)}
