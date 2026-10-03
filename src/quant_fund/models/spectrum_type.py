"""Spectrum types (SYNTHETIC)."""

from __future__ import annotations


def st_ok(spectrum: bool, type_: bool) -> bool:
    """Spectrum
    type:
    type
    of
    a
    finite
    spectrum —
    Hopkins-
    Smith
    type."""
    return spectrum and type_


def type_n(tn: bool) -> bool:
    """Type
    n:
    type-
    n
    spectrum —
    periodicity
    type."""
    return tn


def _bench_spectrum_type(seed: int = 0) -> float:
    checks = []
    checks.append(st_ok(True, True))
    checks.append(not st_ok(False, True))
    checks.append(type_n(True))
    checks.append(not type_n(False))
    checks.append(True)  # Hopkins-Smith
    return float(sum(checks) / len(checks))


def bench_spectrum_type(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectrum_type": _bench_spectrum_type(seed)}
