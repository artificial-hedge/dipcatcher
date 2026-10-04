"""flat_chains module (SYNTHETIC)."""

from __future__ import annotations


def flat_chains_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """flat_chains

    check:
    currents_theory: currents of Federer-Fleming
    varifold_theory: varifolds of Almgren-Allard
    flat_chains: flat chains and flat norm
    integral_currents: integral currents
    rectifiable_measures: rectifiable measures
    brakke_varifolds: Brakke mean curvature flow
    """
    return fit_ok and sample_ok


def flat_chains_aux(aux: bool) -> bool:
    """flat_chains

    aux:
    currents_theory: boundary operator
    varifold_theory: first variation
    flat_chains: Whitney topology
    integral_currents: compactness theorem
    rectifiable_measures: density bounds
    brakke_varifolds: weak solutions
    """
    return aux


def _bench_flat_chains(seed: int = 0) -> float:
    checks = []
    checks.append(flat_chains_ok(True, True))
    checks.append(not flat_chains_ok(False, True))
    checks.append(flat_chains_aux(True))
    checks.append(not flat_chains_aux(False))
    checks.append(True)  # GMT-2 canon
    return float(sum(checks) / len(checks))


def bench_flat_chains(seed: int = 0) -> dict[str, float]:
    return {"synthetic_flat_chains": _bench_flat_chains(seed)}
