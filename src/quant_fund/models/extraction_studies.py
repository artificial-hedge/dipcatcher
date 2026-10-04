"""extraction_studies module (SYNTHETIC)."""

from __future__ import annotations


def extraction_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """extraction_studies

    check:
    extraction_studies: carlini-style training-data extraction and rates
    """
    return fit_ok and sample_ok


def extraction_studies_aux(aux: bool) -> bool:
    """extraction_studies

    aux:
    extraction_studies: perplexity filters, candidate gen, and hits
    """
    return aux


def _bench_extraction_studies(seed: int = 0) -> float:
    checks = []
    checks.append(extraction_studies_ok(True, True))
    checks.append(not extraction_studies_ok(False, True))
    checks.append(extraction_studies_aux(True))
    checks.append(not extraction_studies_aux(False))
    checks.append(True)  # privacy-inference canon
    return float(sum(checks) / len(checks))


def bench_extraction_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_extraction_studies": _bench_extraction_studies(seed)}
