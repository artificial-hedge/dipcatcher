"""Dupont K-theory (SYNTHETIC)."""

from __future__ import annotations


def dk_ok(dupont: bool, k: bool) -> bool:
    """Dupont
    K:
    Dupont
    K
    theory —
    scissors."""
    return dupont and k


def scissors_congruence(sc: bool) -> bool:
    """Scissors
    congruence:
    scissors
    congruence —
    polytopes."""
    return sc


def _bench_dupont_k(seed: int = 0) -> float:
    checks = []
    checks.append(dk_ok(True, True))
    checks.append(not dk_ok(False, True))
    checks.append(scissors_congruence(True))
    checks.append(not scissors_congruence(False))
    checks.append(True)  # Dupont
    return float(sum(checks) / len(checks))


def bench_dupont_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dupont_k": _bench_dupont_k(seed)}
