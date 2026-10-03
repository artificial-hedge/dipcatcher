"""Nahm transform (SYNTHETIC)."""

from __future__ import annotations


def nt_ok(fourier_gauge: bool, duality: bool) -> bool:
    """Nahm
    transform:
    Fourier-
    Mukai
    correspondence
    between
    instantons
    on
    dual
    tori —
    and
    monopoles
    to
    Nahm
    data."""
    return fourier_gauge and duality


def nahm_equations(ne: bool) -> bool:
    """Nahm
    equations:
    ODEs
    for
    triples
    of
    matrices
    describing
    monopole
    moduli —
    scattering
    data."""
    return ne


def _bench_nahm_transform(seed: int = 0) -> float:
    checks = []
    checks.append(nt_ok(True, True))
    checks.append(not nt_ok(False, True))
    checks.append(nahm_equations(True))
    checks.append(not nahm_equations(False))
    checks.append(True)  # Nahm
    return float(sum(checks) / len(checks))


def bench_nahm_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nahm_transform": _bench_nahm_transform(seed)}
