"""shearwater_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def shearwater_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """shearwater_qa_studies

    check:
    shearwater_qa_studies: ShearwaterQA metrics
    """
    return fit_ok and sample_ok


def shearwater_qa_studies_aux(aux: bool) -> bool:
    """shearwater_qa_studies

    aux:
    shearwater_qa_studies: shearwaters, glides, answers, and scores
    """
    return aux


def _bench_shearwater_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(shearwater_qa_studies_ok(True, True))
    checks.append(not shearwater_qa_studies_ok(False, True))
    checks.append(shearwater_qa_studies_aux(True))
    checks.append(not shearwater_qa_studies_aux(False))
    checks.append(True)  # seabird canon
    return float(sum(checks) / len(checks))


def bench_shearwater_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shearwater_qa_studies": _bench_shearwater_qa_studies(seed)}
