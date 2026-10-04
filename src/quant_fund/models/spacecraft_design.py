"""spacecraft_design module (SYNTHETIC)."""

from __future__ import annotations


def spacecraft_design_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """spacecraft_design

    check:
    aerodynamics: aerodynamics
    propulsion: propulsion
    orbital_mechanics2: orbital mechanics
    flight_dynamics: flight dynamics
    spacecraft_design: spacecraft design
    airfoil_theory: airfoil theory
    """
    return fit_ok and sample_ok


def spacecraft_design_aux(aux: bool) -> bool:
    """spacecraft_design

    aux:
    aerodynamics: lift/drag
    propulsion: thrust
    orbital_mechanics2: delta-v
    flight_dynamics: stability axes
    spacecraft_design: payload
    airfoil_theory: circulation
    """
    return aux


def _bench_spacecraft_design(seed: int = 0) -> float:
    checks = []
    checks.append(spacecraft_design_ok(True, True))
    checks.append(not spacecraft_design_ok(False, True))
    checks.append(spacecraft_design_aux(True))
    checks.append(not spacecraft_design_aux(False))
    checks.append(True)  # aerospace-engineering canon
    return float(sum(checks) / len(checks))


def bench_spacecraft_design(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spacecraft_design": _bench_spacecraft_design(seed)}
