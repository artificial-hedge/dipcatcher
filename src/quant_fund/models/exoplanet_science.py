"""exoplanet_science module (SYNTHETIC)."""

from __future__ import annotations


def exoplanet_science_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """exoplanet_science

    check:
    cosmology_2: cosmology
    astrobiology: astrobiology
    astrochemistry: astrochemistry
    helio_seismology: helioseismology
    exoplanet_science: exoplanet science
    galactic_dynamics: galactic dynamics
    """
    return fit_ok and sample_ok


def exoplanet_science_aux(aux: bool) -> bool:
    """exoplanet_science

    aux:
    cosmology_2: expansion and structure
    astrobiology: habitability and life
    astrochemistry: molecules in space
    helio_seismology: solar oscillations
    exoplanet_science: planets beyond
    galactic_dynamics: orbits and mergers
    """
    return aux


def _bench_exoplanet_science(seed: int = 0) -> float:
    checks = []
    checks.append(exoplanet_science_ok(True, True))
    checks.append(not exoplanet_science_ok(False, True))
    checks.append(exoplanet_science_aux(True))
    checks.append(not exoplanet_science_aux(False))
    checks.append(True)  # astronomy-4 canon
    return float(sum(checks) / len(checks))


def bench_exoplanet_science(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exoplanet_science": _bench_exoplanet_science(seed)}
