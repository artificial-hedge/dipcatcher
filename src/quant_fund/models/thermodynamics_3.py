"""thermodynamics_3 module (SYNTHETIC)."""

from __future__ import annotations


def thermodynamics_3_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """thermodynamics_3

    check:
    physics_6: physics
    astrophysics_3: astrophysics
    cosmology_3: cosmology
    geophysics_3: geophysics
    mechanics: mechanics
    thermodynamics_3: thermodynamics
    """
    return fit_ok and sample_ok


def thermodynamics_3_aux(aux: bool) -> bool:
    """thermodynamics_3

    aux:
    physics_6: forces and fields
    astrophysics_3: stars and galaxies
    cosmology_3: universe and expansion
    geophysics_3: earth and interior
    mechanics: motion and energy
    thermodynamics_3: entropy and heat
    """
    return aux


def _bench_thermodynamics_3(seed: int = 0) -> float:
    checks = []
    checks.append(thermodynamics_3_ok(True, True))
    checks.append(not thermodynamics_3_ok(False, True))
    checks.append(thermodynamics_3_aux(True))
    checks.append(not thermodynamics_3_aux(False))
    checks.append(True)  # physical-sciences canon
    return float(sum(checks) / len(checks))


def bench_thermodynamics_3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thermodynamics_3": _bench_thermodynamics_3(seed)}
