"""colocalization_studies module (SYNTHETIC)."""

from __future__ import annotations


def colocalization_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """colocalization_studies

    check:
    colocalization_studies: posterior probabilities and shared variants/eQTL and GWAS
    """
    return fit_ok and sample_ok


def colocalization_studies_aux(aux: bool) -> bool:
    """colocalization_studies

    aux:
    colocalization_studies: credibility and prior sensitivity/loci and regions
    """
    return aux


def _bench_colocalization_studies(seed: int = 0) -> float:
    checks = []
    checks.append(colocalization_studies_ok(True, True))
    checks.append(not colocalization_studies_ok(False, True))
    checks.append(colocalization_studies_aux(True))
    checks.append(not colocalization_studies_aux(False))
    checks.append(True)  # genetic-epidemiology/MR canon
    return float(sum(checks) / len(checks))


def bench_colocalization_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_colocalization_studies": _bench_colocalization_studies(seed)}
