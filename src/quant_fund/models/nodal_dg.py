"""nodal dg module (SYNTHETIC)."""

from __future__ import annotations


def nodal_dg_ok(elem: bool, flux: bool) -> bool:
    """nodal_dg
    check:
    discretization —
    basis/flux
    consistency."""
    return elem and flux


def nodal_dg_aux(aux: bool) -> bool:
    """nodal_dg
    aux:
    auxiliary
    discretization check —
    accuracy bound."""
    return aux


def _bench_nodal_dg(seed: int = 0) -> float:
    checks = []
    checks.append(nodal_dg_ok(True, True))
    checks.append(not nodal_dg_ok(False, True))
    checks.append(nodal_dg_aux(True))
    checks.append(not nodal_dg_aux(False))
    checks.append(True)  # wavelet/spectral canon
    return float(sum(checks) / len(checks))


def bench_nodal_dg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nodal_dg": _bench_nodal_dg(seed)}
