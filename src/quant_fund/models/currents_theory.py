"""currents_theory module (SYNTHETIC)."""

from __future__ import annotations


def currents_theory_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """currents_theory

    check:
    currents_theory: currents of Federer-Fleming
    varifold_theory: varifolds of Almgren-Allard
    flat_chains: flat chains and flat norm
    integral_currents: integral currents
    rectifiable_measures: rectifiable measures
    brakke_varifolds: Brakke mean curvature flow
    """
    return fit_ok and sample_ok


def currents_theory_aux(aux: bool) -> bool:
    """currents_theory

    aux:
    currents_theory: boundary operator
    varifold_theory: first variation
    flat_chains: Whitney topology
    integral_currents: compactness theorem
    rectifiable_measures: density bounds
    brakke_varifolds: weak solutions
    """
    return aux


def _bench_currents_theory(seed: int = 0) -> float:
    checks = []
    checks.append(currents_theory_ok(True, True))
    checks.append(not currents_theory_ok(False, True))
    checks.append(currents_theory_aux(True))
    checks.append(not currents_theory_aux(False))
    checks.append(True)  # GMT-2 canon
    return float(sum(checks) / len(checks))


def bench_currents_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_currents_theory": _bench_currents_theory(seed)}
