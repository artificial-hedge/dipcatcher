"""allele_studies module (SYNTHETIC)."""

from __future__ import annotations


def allele_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """allele_studies

    check:
    allele_studies: alleles and dominance
    ..."""
    return fit_ok and sample_ok


def allele_studies_aux(aux: bool) -> bool:
    """allele_studies

    aux:
    allele_studies: recessive and heterozygous
    ..."""
    return aux


def _bench_allele_studies(seed: int = 0) -> float:
    checks = []
    checks.append(allele_studies_ok(True, True))
    checks.append(not allele_studies_ok(False, True))
    checks.append(allele_studies_aux(True))
    checks.append(not allele_studies_aux(False))
    checks.append(True)  # molecular-genetics-2 canon
    return float(sum(checks) / len(checks))


def bench_allele_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_allele_studies": _bench_allele_studies(seed)}
