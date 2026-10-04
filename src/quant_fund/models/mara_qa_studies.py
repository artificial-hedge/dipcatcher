"""mara_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mara_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mara_qa_studies

    check:
    mara_qa_studies: MaraQA metrics
    """
    return fit_ok and sample_ok


def mara_qa_studies_aux(aux: bool) -> bool:
    """mara_qa_studies

    aux:
    mara_qa_studies: maras, patagonian scrub, answers, and scores
    """
    return aux


def _bench_mara_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mara_qa_studies_ok(True, True))
    checks.append(not mara_qa_studies_ok(False, True))
    checks.append(mara_qa_studies_aux(True))
    checks.append(not mara_qa_studies_aux(False))
    checks.append(True)  # small-mammal-2 canon
    return float(sum(checks) / len(checks))


def bench_mara_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mara_qa_studies": _bench_mara_qa_studies(seed)}
