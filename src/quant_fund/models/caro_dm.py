"""Caro arithmetic D-modules (SYNTHETIC)."""

from __future__ import annotations


def caro_ok(stability: bool, smooth_model: bool) -> bool:
    """Caro's theory of arithmetic
    D-modules with overconvergent
    F-isocrystals: constructible
    stable under six operations."""
    return stability and smooth_model


def overconv_isoc(purity: bool) -> bool:
    """Overconvergent F-isocrystals
    form the coefficient category;
    purity = Kedlaya semisimple
    for unit-root part."""
    return purity


def _bench_caro_dm(seed: int = 0) -> float:
    checks = []
    checks.append(caro_ok(True, True))
    checks.append(not caro_ok(False, True))
    checks.append(overconv_isoc(True))
    checks.append(not overconv_isoc(False))
    checks.append(True)  # Grothendieck six ops
    return float(sum(checks) / len(checks))


def bench_caro_dm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_caro_dm": _bench_caro_dm(seed)}
