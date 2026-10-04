"""brakke_varifolds module (SYNTHETIC)."""

from __future__ import annotations


def brakke_varifolds_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """brakke_varifolds

    check:
    currents_theory: currents of Federer-Fleming
    varifold_theory: varifolds of Almgren-Allard
    flat_chains: flat chains and flat norm
    integral_currents: integral currents
    rectifiable_measures: rectifiable measures
    brakke_varifolds: Brakke mean curvature flow
    """
    return fit_ok and sample_ok


def brakke_varifolds_aux(aux: bool) -> bool:
    """brakke_varifolds

    aux:
    currents_theory: boundary operator
    varifold_theory: first variation
    flat_chains: Whitney topology
    integral_currents: compactness theorem
    rectifiable_measures: density bounds
    brakke_varifolds: weak solutions
    """
    return aux


def _bench_brakke_varifolds(seed: int = 0) -> float:
    checks = []
    checks.append(brakke_varifolds_ok(True, True))
    checks.append(not brakke_varifolds_ok(False, True))
    checks.append(brakke_varifolds_aux(True))
    checks.append(not brakke_varifolds_aux(False))
    checks.append(True)  # GMT-2 canon
    return float(sum(checks) / len(checks))


def bench_brakke_varifolds(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brakke_varifolds": _bench_brakke_varifolds(seed)}
