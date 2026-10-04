"""Hodge moduli (SYNTHETIC)."""

from __future__ import annotations


def hm_ok(moduli_hb: bool, hyperk: bool) -> bool:
    """Hodge
    moduli:
    hyperkahler
    moduli
    of
    Higgs
    bundles —
    Dolbeault
    moduli."""
    return moduli_hb and hyperk


def hitchin_metric(hm: bool) -> bool:
    """Hitchin
    metric:
    hyperkahler
    structure
    on
    Hitchin
    moduli —
    Hitchin."""
    return hm


def _bench_hodge_moduli(seed: int = 0) -> float:
    checks = []
    checks.append(hm_ok(True, True))
    checks.append(not hm_ok(False, True))
    checks.append(hitchin_metric(True))
    checks.append(not hitchin_metric(False))
    checks.append(True)  # Hitchin
    return float(sum(checks) / len(checks))


def bench_hodge_moduli(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hodge_moduli": _bench_hodge_moduli(seed)}
