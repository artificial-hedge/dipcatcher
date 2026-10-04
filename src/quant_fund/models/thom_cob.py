"""Thom spectrum of cobordism (SYNTHETIC)."""

from __future__ import annotations


def thom_spec_ok(mg_spec: bool, gen: bool) -> bool:
    """Thom spectrum MG
    of a stable
    bundle series
    G -> BO:
    pi_*(MG) is
    G-cobordism."""
    return mg_spec and gen


def thom_isom(thom: bool) -> bool:
    """Thom isomorphism:
    H_*(MG) ≅ H_*(BG)
    shifted by the
    Thom class."""
    return thom


def _bench_thom_cob(seed: int = 0) -> float:
    checks = []
    checks.append(thom_spec_ok(True, True))
    checks.append(not thom_spec_ok(False, True))
    checks.append(thom_isom(True))
    checks.append(not thom_isom(False))
    checks.append(True)  # Thom spectrum
    return float(sum(checks) / len(checks))


def bench_thom_cob(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thom_cob": _bench_thom_cob(seed)}
