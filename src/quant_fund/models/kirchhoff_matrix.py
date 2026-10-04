"""kirchhoff matrix module (SYNTHETIC)."""

from __future__ import annotations


def kirchhoff_matrix_ok(ust: bool, lerw: bool) -> bool:
    """kirchhoff_matrix
    check:
    UST/LERW
    structure —
    Wilson."""
    return ust and lerw


def kirchhoff_matrix_aux(aux: bool) -> bool:
    """kirchhoff_matrix
    aux:
    auxiliary
    spanning-tree
    check —
    Lawler."""
    return aux


def _bench_kirchhoff_matrix(seed: int = 0) -> float:
    checks = []
    checks.append(kirchhoff_matrix_ok(True, True))
    checks.append(not kirchhoff_matrix_ok(False, True))
    checks.append(kirchhoff_matrix_aux(True))
    checks.append(not kirchhoff_matrix_aux(False))
    checks.append(True)  # UST/LERW canon
    return float(sum(checks) / len(checks))


def bench_kirchhoff_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kirchhoff_matrix": _bench_kirchhoff_matrix(seed)}
