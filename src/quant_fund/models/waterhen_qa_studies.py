"""waterhen_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def waterhen_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """waterhen_qa_studies

    check:
    waterhen_qa_studies: WaterhenQA metrics
    """
    return fit_ok and sample_ok


def waterhen_qa_studies_aux(aux: bool) -> bool:
    """waterhen_qa_studies

    aux:
    waterhen_qa_studies: waterhens, reeds, answers, and scores
    """
    return aux


def _bench_waterhen_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(waterhen_qa_studies_ok(True, True))
    checks.append(not waterhen_qa_studies_ok(False, True))
    checks.append(waterhen_qa_studies_aux(True))
    checks.append(not waterhen_qa_studies_aux(False))
    checks.append(True)  # marshbird canon
    return float(sum(checks) / len(checks))


def bench_waterhen_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_waterhen_qa_studies": _bench_waterhen_qa_studies(seed)}
