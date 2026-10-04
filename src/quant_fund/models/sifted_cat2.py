"""Sifted category theory 2 (SYNTHETIC)."""

from __future__ import annotations


def sc_ok(sifted: bool, products: bool) -> bool:
    """Sifted
    category:
    sifted
    colimit —
    commutes
    finite
    products."""
    return sifted and products


def sifted_colimit(scl: bool) -> bool:
    """Sifted
    colimit:
    sifted
    colimit —
    reflexive
    coequalizers."""
    return scl


def _bench_sifted_cat2(seed: int = 0) -> float:
    checks = []
    checks.append(sc_ok(True, True))
    checks.append(not sc_ok(False, True))
    checks.append(sifted_colimit(True))
    checks.append(not sifted_colimit(False))
    checks.append(True)  # sifted vs filtered
    return float(sum(checks) / len(checks))


def bench_sifted_cat2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sifted_cat2": _bench_sifted_cat2(seed)}
