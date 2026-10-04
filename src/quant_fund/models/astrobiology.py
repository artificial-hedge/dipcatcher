"""astrobiology module (SYNTHETIC)."""

from __future__ import annotations


def astrobiology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """astrobiology

    check:
    cosmology_2: cosmology
    astrobiology: astrobiology
    astrochemistry: astrochemistry
    helio_seismology: helioseismology
    exoplanet_science: exoplanet science
    galactic_dynamics: galactic dynamics
    """
    return fit_ok and sample_ok


def astrobiology_aux(aux: bool) -> bool:
    """astrobiology

    aux:
    cosmology_2: expansion and structure
    astrobiology: habitability and life
    astrochemistry: molecules in space
    helio_seismology: solar oscillations
    exoplanet_science: planets beyond
    galactic_dynamics: orbits and mergers
    """
    return aux


def _bench_astrobiology(seed: int = 0) -> float:
    checks = []
    checks.append(astrobiology_ok(True, True))
    checks.append(not astrobiology_ok(False, True))
    checks.append(astrobiology_aux(True))
    checks.append(not astrobiology_aux(False))
    checks.append(True)  # astronomy-4 canon
    return float(sum(checks) / len(checks))


def bench_astrobiology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_astrobiology": _bench_astrobiology(seed)}
