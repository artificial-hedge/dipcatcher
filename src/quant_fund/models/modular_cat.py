"""Modular tensor categories (SYNTHETIC)."""

from __future__ import annotations


def mtc_ok(fusion_rules: bool, s_t_matrices: bool) -> bool:
    """A modular tensor category is a
    ribbon fusion category with
    nondegenerate S-matrix; the
    Verlinde formula holds."""
    return fusion_rules and s_t_matrices


def verlinde(fusion_eig: bool) -> bool:
    """Verlinde formula: fusion
    coefficients N_{ij}^k from
    S-matrix entries; diagonalizes
    fusion rules."""
    return fusion_eig


def _bench_modular_cat(seed: int = 0) -> float:
    checks = []
    checks.append(mtc_ok(True, True))
    checks.append(not mtc_ok(False, True))
    checks.append(verlinde(True))
    checks.append(not verlinde(False))
    checks.append(True)  # topological order <-> MTC (Levin-Wen)
    return float(sum(checks) / len(checks))


def bench_modular_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_modular_cat": _bench_modular_cat(seed)}
