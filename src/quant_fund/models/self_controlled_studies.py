"""self_controlled_studies module (SYNTHETIC)."""

from __future__ import annotations


def self_controlled_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """self_controlled_studies

    check:
    self_controlled_studies: case series and crossover/within-person and windows
    """
    return fit_ok and sample_ok


def self_controlled_studies_aux(aux: bool) -> bool:
    """self_controlled_studies

    aux:
    self_controlled_studies: risk and reference periods/rates and SCCS
    """
    return aux


def _bench_self_controlled_studies(seed: int = 0) -> float:
    checks = []
    checks.append(self_controlled_studies_ok(True, True))
    checks.append(not self_controlled_studies_ok(False, True))
    checks.append(self_controlled_studies_aux(True))
    checks.append(not self_controlled_studies_aux(False))
    checks.append(True)  # causal-RWE-2 canon
    return float(sum(checks) / len(checks))


def bench_self_controlled_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_self_controlled_studies": _bench_self_controlled_studies(seed)}
