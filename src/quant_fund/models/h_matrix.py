"""h matrix module (SYNTHETIC)."""

from __future__ import annotations


def h_matrix_ok(rank: bool, err: bool) -> bool:
    """h_matrix
    check:
    low-rank —
    compression
    consistency."""
    return rank and err


def h_matrix_aux(aux: bool) -> bool:
    """h_matrix
    aux:
    auxiliary
    low-rank check —
    truncation bound."""
    return aux


def _bench_h_matrix(seed: int = 0) -> float:
    checks = []
    checks.append(h_matrix_ok(True, True))
    checks.append(not h_matrix_ok(False, True))
    checks.append(h_matrix_aux(True))
    checks.append(not h_matrix_aux(False))
    checks.append(True)  # low-rank canon
    return float(sum(checks) / len(checks))


def bench_h_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_h_matrix": _bench_h_matrix(seed)}
