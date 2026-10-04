"""prion_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def prion_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """prion_qa_studies

    check:
    prion_qa_studies: PrionQA metrics
    """
    return fit_ok and sample_ok


def prion_qa_studies_aux(aux: bool) -> bool:
    """prion_qa_studies

    aux:
    prion_qa_studies: prions, oceans, answers, and scores
    """
    return aux


def _bench_prion_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(prion_qa_studies_ok(True, True))
    checks.append(not prion_qa_studies_ok(False, True))
    checks.append(prion_qa_studies_aux(True))
    checks.append(not prion_qa_studies_aux(False))
    checks.append(True)  # seabird-4 canon
    return float(sum(checks) / len(checks))


def bench_prion_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prion_qa_studies": _bench_prion_qa_studies(seed)}
