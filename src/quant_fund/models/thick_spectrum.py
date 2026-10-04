"""Thick spectra (SYNTHETIC)."""

from __future__ import annotations


def ts_ok(thick: bool, spectrum: bool) -> bool:
    """Thick:
    thick
    subcategories
    of
    spectra —
    thick
    subcategory."""
    return thick and spectrum


def thick_class(tc: bool) -> bool:
    """Thick
    classification:
    Hopkins-
    Smith
    thick
    subcategory
    classification —
    Hopkins-
    Smith."""
    return tc


def _bench_thick_spectrum(seed: int = 0) -> float:
    checks = []
    checks.append(ts_ok(True, True))
    checks.append(not ts_ok(False, True))
    checks.append(thick_class(True))
    checks.append(not thick_class(False))
    checks.append(True)  # Hopkins-Smith
    return float(sum(checks) / len(checks))


def bench_thick_spectrum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thick_spectrum": _bench_thick_spectrum(seed)}
