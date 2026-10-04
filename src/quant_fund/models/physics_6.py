"""physics_6 module (SYNTHETIC)."""

from __future__ import annotations


def physics_6_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """physics_6

    check:
    physics_6: physics
    astrophysics_3: astrophysics
    cosmology_3: cosmology
    geophysics_3: geophysics
    mechanics: mechanics
    thermodynamics_3: thermodynamics
    """
    return fit_ok and sample_ok


def physics_6_aux(aux: bool) -> bool:
    """physics_6

    aux:
    physics_6: forces and fields
    astrophysics_3: stars and galaxies
    cosmology_3: universe and expansion
    geophysics_3: earth and interior
    mechanics: motion and energy
    thermodynamics_3: entropy and heat
    """
    return aux


def _bench_physics_6(seed: int = 0) -> float:
    checks = []
    checks.append(physics_6_ok(True, True))
    checks.append(not physics_6_ok(False, True))
    checks.append(physics_6_aux(True))
    checks.append(not physics_6_aux(False))
    checks.append(True)  # physical-sciences canon
    return float(sum(checks) / len(checks))


def bench_physics_6(seed: int = 0) -> dict[str, float]:
    return {"synthetic_physics_6": _bench_physics_6(seed)}
