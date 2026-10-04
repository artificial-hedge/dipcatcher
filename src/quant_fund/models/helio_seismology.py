"""helio_seismology module (SYNTHETIC)."""

from __future__ import annotations


def helio_seismology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """helio_seismology

    check:
    cosmology_2: cosmology
    astrobiology: astrobiology
    astrochemistry: astrochemistry
    helio_seismology: helioseismology
    exoplanet_science: exoplanet science
    galactic_dynamics: galactic dynamics
    """
    return fit_ok and sample_ok


def helio_seismology_aux(aux: bool) -> bool:
    """helio_seismology

    aux:
    cosmology_2: expansion and structure
    astrobiology: habitability and life
    astrochemistry: molecules in space
    helio_seismology: solar oscillations
    exoplanet_science: planets beyond
    galactic_dynamics: orbits and mergers
    """
    return aux


def _bench_helio_seismology(seed: int = 0) -> float:
    checks = []
    checks.append(helio_seismology_ok(True, True))
    checks.append(not helio_seismology_ok(False, True))
    checks.append(helio_seismology_aux(True))
    checks.append(not helio_seismology_aux(False))
    checks.append(True)  # astronomy-4 canon
    return float(sum(checks) / len(checks))


def bench_helio_seismology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_helio_seismology": _bench_helio_seismology(seed)}
