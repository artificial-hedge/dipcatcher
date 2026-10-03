"""rectifiable_measures module (SYNTHETIC)."""

from __future__ import annotations


def rectifiable_measures_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rectifiable_measures

    check:
    currents_theory: currents of Federer-Fleming
    varifold_theory: varifolds of Almgren-Allard
    flat_chains: flat chains and flat norm
    integral_currents: integral currents
    rectifiable_measures: rectifiable measures
    brakke_varifolds: Brakke mean curvature flow
    """
    return fit_ok and sample_ok


def rectifiable_measures_aux(aux: bool) -> bool:
    """rectifiable_measures

    aux:
    currents_theory: boundary operator
    varifold_theory: first variation
    flat_chains: Whitney topology
    integral_currents: compactness theorem
    rectifiable_measures: density bounds
    brakke_varifolds: weak solutions
    """
    return aux


def _bench_rectifiable_measures(seed: int = 0) -> float:
    checks = []
    checks.append(rectifiable_measures_ok(True, True))
    checks.append(not rectifiable_measures_ok(False, True))
    checks.append(rectifiable_measures_aux(True))
    checks.append(not rectifiable_measures_aux(False))
    checks.append(True)  # GMT-2 canon
    return float(sum(checks) / len(checks))


def bench_rectifiable_measures(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rectifiable_measures": _bench_rectifiable_measures(seed)}
