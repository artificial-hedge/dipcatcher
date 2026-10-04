"""lead_optimization_studies module (SYNTHETIC)."""

from __future__ import annotations


def lead_optimization_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lead_optimization_studies

    check:
    lead_optimization_studies: potency and selectivity/profile and optimization
    """
    return fit_ok and sample_ok


def lead_optimization_studies_aux(aux: bool) -> bool:
    """lead_optimization_studies

    aux:
    lead_optimization_studies: analog and sar/improvement and multiparameter
    """
    return aux


def _bench_lead_optimization_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lead_optimization_studies_ok(True, True))
    checks.append(not lead_optimization_studies_ok(False, True))
    checks.append(lead_optimization_studies_aux(True))
    checks.append(not lead_optimization_studies_aux(False))
    checks.append(True)  # drug-discovery canon
    return float(sum(checks) / len(checks))


def bench_lead_optimization_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lead_optimization_studies": _bench_lead_optimization_studies(seed)}
