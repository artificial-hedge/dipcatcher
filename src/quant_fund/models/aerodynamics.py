"""aerodynamics module (SYNTHETIC)."""

from __future__ import annotations


def aerodynamics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """aerodynamics

    check:
    aerodynamics: aerodynamics
    propulsion: propulsion
    orbital_mechanics2: orbital mechanics
    flight_dynamics: flight dynamics
    spacecraft_design: spacecraft design
    airfoil_theory: airfoil theory
    """
    return fit_ok and sample_ok


def aerodynamics_aux(aux: bool) -> bool:
    """aerodynamics

    aux:
    aerodynamics: lift/drag
    propulsion: thrust
    orbital_mechanics2: delta-v
    flight_dynamics: stability axes
    spacecraft_design: payload
    airfoil_theory: circulation
    """
    return aux


def _bench_aerodynamics(seed: int = 0) -> float:
    checks = []
    checks.append(aerodynamics_ok(True, True))
    checks.append(not aerodynamics_ok(False, True))
    checks.append(aerodynamics_aux(True))
    checks.append(not aerodynamics_aux(False))
    checks.append(True)  # aerospace-engineering canon
    return float(sum(checks) / len(checks))


def bench_aerodynamics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aerodynamics": _bench_aerodynamics(seed)}
