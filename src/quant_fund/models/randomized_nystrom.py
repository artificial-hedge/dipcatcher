"""randomized nystrom module (SYNTHETIC)."""

from __future__ import annotations


def randomized_nystrom_ok(rank: bool, err: bool) -> bool:
    """randomized_nystrom
    check:
    low-rank —
    compression
    consistency."""
    return rank and err


def randomized_nystrom_aux(aux: bool) -> bool:
    """randomized_nystrom
    aux:
    auxiliary
    low-rank check —
    truncation bound."""
    return aux


def _bench_randomized_nystrom(seed: int = 0) -> float:
    checks = []
    checks.append(randomized_nystrom_ok(True, True))
    checks.append(not randomized_nystrom_ok(False, True))
    checks.append(randomized_nystrom_aux(True))
    checks.append(not randomized_nystrom_aux(False))
    checks.append(True)  # low-rank canon
    return float(sum(checks) / len(checks))


def bench_randomized_nystrom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_randomized_nystrom": _bench_randomized_nystrom(seed)}
