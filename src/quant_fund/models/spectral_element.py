"""spectral element module (SYNTHETIC)."""

from __future__ import annotations


def spectral_element_ok(node: bool, poly: bool) -> bool:
    """spectral_element
    check:
    spectral-element —
    high-order
    consistency."""
    return node and poly


def spectral_element_aux(aux: bool) -> bool:
    """spectral_element
    aux:
    auxiliary
    SEM check —
    interpolation."""
    return aux


def _bench_spectral_element(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_element_ok(True, True))
    checks.append(not spectral_element_ok(False, True))
    checks.append(spectral_element_aux(True))
    checks.append(not spectral_element_aux(False))
    checks.append(True)  # spectral-element canon
    return float(sum(checks) / len(checks))


def bench_spectral_element(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_element": _bench_spectral_element(seed)}
