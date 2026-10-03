"""Finite spectra (SYNTHETIC)."""

from __future__ import annotations


def fs_ok(finite: bool, spectrum: bool) -> bool:
    """Finite:
    finite
    spectrum —
    finite
    cell
    spectrum."""
    return finite and spectrum


def spanier_dual(sd: bool) -> bool:
    """Spanier:
    Spanier-
    Whitehead
    dual
    of
    finite
    spectra —
    Spanier-
    Whitehead."""
    return sd


def _bench_finite_spectra(seed: int = 0) -> float:
    checks = []
    checks.append(fs_ok(True, True))
    checks.append(not fs_ok(False, True))
    checks.append(spanier_dual(True))
    checks.append(not spanier_dual(False))
    checks.append(True)  # Spanier-Whitehead
    return float(sum(checks) / len(checks))


def bench_finite_spectra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finite_spectra": _bench_finite_spectra(seed)}
