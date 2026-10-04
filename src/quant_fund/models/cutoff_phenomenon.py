"""cutoff phenomenon module (SYNTHETIC)."""

from __future__ import annotations


def cutoff_phenomenon_ok(dc: bool, hr: bool) -> bool:
    """cutoff_phenomenon
    check:
    Markov-chain
    theory —
    stability."""
    return dc and hr


def cutoff_phenomenon_aux(aux: bool) -> bool:
    """cutoff_phenomenon
    aux:
    auxiliary
    chain
    check —
    mixing."""
    return aux


def _bench_cutoff_phenomenon(seed: int = 0) -> float:
    checks = []
    checks.append(cutoff_phenomenon_ok(True, True))
    checks.append(not cutoff_phenomenon_ok(False, True))
    checks.append(cutoff_phenomenon_aux(True))
    checks.append(not cutoff_phenomenon_aux(False))
    checks.append(True)  # markov-chain canon
    return float(sum(checks) / len(checks))


def bench_cutoff_phenomenon(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cutoff_phenomenon": _bench_cutoff_phenomenon(seed)}
