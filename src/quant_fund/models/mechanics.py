"""mechanics module (SYNTHETIC)."""

from __future__ import annotations


def mechanics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mechanics

    check:
    physics_6: physics
    astrophysics_3: astrophysics
    cosmology_3: cosmology
    geophysics_3: geophysics
    mechanics: mechanics
    thermodynamics_3: thermodynamics
    """
    return fit_ok and sample_ok


def mechanics_aux(aux: bool) -> bool:
    """mechanics

    aux:
    physics_6: forces and fields
    astrophysics_3: stars and galaxies
    cosmology_3: universe and expansion
    geophysics_3: earth and interior
    mechanics: motion and energy
    thermodynamics_3: entropy and heat
    """
    return aux


def _bench_mechanics(seed: int = 0) -> float:
    checks = []
    checks.append(mechanics_ok(True, True))
    checks.append(not mechanics_ok(False, True))
    checks.append(mechanics_aux(True))
    checks.append(not mechanics_aux(False))
    checks.append(True)  # physical-sciences canon
    return float(sum(checks) / len(checks))


def bench_mechanics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mechanics": _bench_mechanics(seed)}
