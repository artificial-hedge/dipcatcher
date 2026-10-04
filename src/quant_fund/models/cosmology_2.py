"""cosmology_2 module (SYNTHETIC)."""

from __future__ import annotations


def cosmology_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cosmology_2

    check:
    cosmology_2: cosmology
    astrobiology: astrobiology
    astrochemistry: astrochemistry
    helio_seismology: helioseismology
    exoplanet_science: exoplanet science
    galactic_dynamics: galactic dynamics
    """
    return fit_ok and sample_ok


def cosmology_2_aux(aux: bool) -> bool:
    """cosmology_2

    aux:
    cosmology_2: expansion and structure
    astrobiology: habitability and life
    astrochemistry: molecules in space
    helio_seismology: solar oscillations
    exoplanet_science: planets beyond
    galactic_dynamics: orbits and mergers
    """
    return aux


def _bench_cosmology_2(seed: int = 0) -> float:
    checks = []
    checks.append(cosmology_2_ok(True, True))
    checks.append(not cosmology_2_ok(False, True))
    checks.append(cosmology_2_aux(True))
    checks.append(not cosmology_2_aux(False))
    checks.append(True)  # astronomy-4 canon
    return float(sum(checks) / len(checks))


def bench_cosmology_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cosmology_2": _bench_cosmology_2(seed)}
