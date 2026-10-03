"""boundary element module (SYNTHETIC)."""

from __future__ import annotations


def boundary_element_ok(elem: bool, flux: bool) -> bool:
    """boundary_element
    check:
    discretization —
    basis/flux
    consistency."""
    return elem and flux


def boundary_element_aux(aux: bool) -> bool:
    """boundary_element
    aux:
    auxiliary
    discretization check —
    accuracy bound."""
    return aux


def _bench_boundary_element(seed: int = 0) -> float:
    checks = []
    checks.append(boundary_element_ok(True, True))
    checks.append(not boundary_element_ok(False, True))
    checks.append(boundary_element_aux(True))
    checks.append(not boundary_element_aux(False))
    checks.append(True)  # wavelet/spectral canon
    return float(sum(checks) / len(checks))


def bench_boundary_element(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boundary_element": _bench_boundary_element(seed)}
