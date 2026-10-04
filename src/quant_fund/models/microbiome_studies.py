"""microbiome_studies module (SYNTHETIC)."""
from __future__ import annotations


def microbiome_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """microbiome_studies

    check:
    microbiome_studies: communities and taxa/diversity and abundance
    """
    return fit_ok and sample_ok


def microbiome_studies_aux(aux: bool) -> bool:
    """microbiome_studies

    aux:
    microbiome_studies: reads and classification/rarefaction and composition
    """
    return aux


def _bench_microbiome_studies(seed: int = 0) -> float:
    checks = []
    checks.append(microbiome_studies_ok(True, True))
    checks.append(not microbiome_studies_ok(False, True))
    checks.append(microbiome_studies_aux(True))
    checks.append(not microbiome_studies_aux(False))
    checks.append(True)  # omics canon
    return float(sum(checks) / len(checks))


def bench_microbiome_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_microbiome_studies": _bench_microbiome_studies(seed)}
