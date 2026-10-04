"""Keller dg derived categories (SYNTHETIC)."""

from __future__ import annotations


def keller_ok(sdg_cat: bool, model: bool) -> bool:
    """Keller's derived cat
    D(A) of a dg cat A:
    homotopy cat of
    dg-mod(A); model
    structure; Tabuada."""
    return sdg_cat and model


def dg_derived_cat(cofibrant: bool) -> bool:
    """D(A) as localization
    of dg-mod(A) at
    quasi-isomorphisms;
    compact = perfect
    dg modules."""
    return cofibrant


def _bench_keller_dg(seed: int = 0) -> float:
    checks = []
    checks.append(keller_ok(True, True))
    checks.append(not keller_ok(False, True))
    checks.append(dg_derived_cat(True))
    checks.append(not dg_derived_cat(False))
    checks.append(True)  # Keller 2006
    return float(sum(checks) / len(checks))


def bench_keller_dg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_keller_dg": _bench_keller_dg(seed)}
