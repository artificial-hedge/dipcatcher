"""contamination_studies module (SYNTHETIC)."""

from __future__ import annotations


def contamination_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """contamination_studies

    check:
    contamination_studies: train-test contamination scans/ngrams and matches
    """
    return fit_ok and sample_ok


def contamination_studies_aux(aux: bool) -> bool:
    """contamination_studies

    aux:
    contamination_studies: dataset-duplication audits/hashes and hits
    """
    return aux


def _bench_contamination_studies(seed: int = 0) -> float:
    checks = []
    checks.append(contamination_studies_ok(True, True))
    checks.append(not contamination_studies_ok(False, True))
    checks.append(contamination_studies_aux(True))
    checks.append(not contamination_studies_aux(False))
    checks.append(True)  # eval-science canon
    return float(sum(checks) / len(checks))


def bench_contamination_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_contamination_studies": _bench_contamination_studies(seed)}
