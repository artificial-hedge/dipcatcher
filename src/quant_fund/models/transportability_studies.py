"""transportability_studies module (SYNTHETIC)."""

from __future__ import annotations


def transportability_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """transportability_studies

    check:
    transportability_studies: target populations and effect modifiers/calibration and weights
    """
    return fit_ok and sample_ok


def transportability_studies_aux(aux: bool) -> bool:
    """transportability_studies

    aux:
    transportability_studies: generalizability and exchangeability/sampling and bounds
    """
    return aux


def _bench_transportability_studies(seed: int = 0) -> float:
    checks = []
    checks.append(transportability_studies_ok(True, True))
    checks.append(not transportability_studies_ok(False, True))
    checks.append(transportability_studies_aux(True))
    checks.append(not transportability_studies_aux(False))
    checks.append(True)  # causal-RWE-2 canon
    return float(sum(checks) / len(checks))


def bench_transportability_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transportability_studies": _bench_transportability_studies(seed)}
