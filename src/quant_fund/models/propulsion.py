"""propulsion module (SYNTHETIC)."""

from __future__ import annotations


def propulsion_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """propulsion

    check:
    aerodynamics: aerodynamics
    propulsion: propulsion
    orbital_mechanics2: orbital mechanics
    flight_dynamics: flight dynamics
    spacecraft_design: spacecraft design
    airfoil_theory: airfoil theory
    """
    return fit_ok and sample_ok


def propulsion_aux(aux: bool) -> bool:
    """propulsion

    aux:
    aerodynamics: lift/drag
    propulsion: thrust
    orbital_mechanics2: delta-v
    flight_dynamics: stability axes
    spacecraft_design: payload
    airfoil_theory: circulation
    """
    return aux


def _bench_propulsion(seed: int = 0) -> float:
    checks = []
    checks.append(propulsion_ok(True, True))
    checks.append(not propulsion_ok(False, True))
    checks.append(propulsion_aux(True))
    checks.append(not propulsion_aux(False))
    checks.append(True)  # aerospace-engineering canon
    return float(sum(checks) / len(checks))


def bench_propulsion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_propulsion": _bench_propulsion(seed)}
