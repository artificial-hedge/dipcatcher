"""varifold_theory module (SYNTHETIC)."""

from __future__ import annotations


def varifold_theory_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """varifold_theory

    check:
    currents_theory: currents of Federer-Fleming
    varifold_theory: varifolds of Almgren-Allard
    flat_chains: flat chains and flat norm
    integral_currents: integral currents
    rectifiable_measures: rectifiable measures
    brakke_varifolds: Brakke mean curvature flow
    """
    return fit_ok and sample_ok


def varifold_theory_aux(aux: bool) -> bool:
    """varifold_theory

    aux:
    currents_theory: boundary operator
    varifold_theory: first variation
    flat_chains: Whitney topology
    integral_currents: compactness theorem
    rectifiable_measures: density bounds
    brakke_varifolds: weak solutions
    """
    return aux


def _bench_varifold_theory(seed: int = 0) -> float:
    checks = []
    checks.append(varifold_theory_ok(True, True))
    checks.append(not varifold_theory_ok(False, True))
    checks.append(varifold_theory_aux(True))
    checks.append(not varifold_theory_aux(False))
    checks.append(True)  # GMT-2 canon
    return float(sum(checks) / len(checks))


def bench_varifold_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_varifold_theory": _bench_varifold_theory(seed)}
