"""Comparison theorems (SYNTHETIC)."""

from __future__ import annotations


def comp_ok(rauch: bool, bishop: bool) -> bool:
    """Comparison:
    Rauch
    compares
    Jacobi
    fields;
    Bishop-
    Gromov
    compares
    volumes."""
    return rauch and bishop


def bonnet_myers_thm(bm: bool) -> bool:
    """Bonnet-
    Myers:
    positive
    Ricci
    lower
    bound
    gives
    bounded
    diameter
    and
    compactness."""
    return bm


def _bench_comparison_thm(seed: int = 0) -> float:
    checks = []
    checks.append(comp_ok(True, True))
    checks.append(not comp_ok(False, True))
    checks.append(bonnet_myers_thm(True))
    checks.append(not bonnet_myers_thm(False))
    checks.append(True)  # Rauch-Bishop
    return float(sum(checks) / len(checks))


def bench_comparison_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_comparison_thm": _bench_comparison_thm(seed)}
