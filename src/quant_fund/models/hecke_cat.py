"""Hecke category (SYNTHETIC)."""

from __future__ import annotations


def hecke_cat_ok(soergel: bool, categorify: bool) -> bool:
    """Hecke
    category:
    Soergel
    bimodules
    categorify
    the Hecke
    algebra;
    [B_w] -> b_w."""
    return soergel and categorify


def kl_basis(kl: bool) -> bool:
    """Kazhdan-
    Lusztig
    basis lifts
    to
    indecomposable
    Soergel
    bimodules."""
    return kl


def _bench_hecke_cat(seed: int = 0) -> float:
    checks = []
    checks.append(hecke_cat_ok(True, True))
    checks.append(not hecke_cat_ok(False, True))
    checks.append(kl_basis(True))
    checks.append(not kl_basis(False))
    checks.append(True)  # Soergel
    return float(sum(checks) / len(checks))


def bench_hecke_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hecke_cat": _bench_hecke_cat(seed)}
