"""oribi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def oribi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """oribi_qa_studies

    check:
    oribi_qa_studies: OribiQA metrics
    """
    return fit_ok and sample_ok


def oribi_qa_studies_aux(aux: bool) -> bool:
    """oribi_qa_studies

    aux:
    oribi_qa_studies: oribis, highveld grass, answers, and scores
    """
    return aux


def _bench_oribi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(oribi_qa_studies_ok(True, True))
    checks.append(not oribi_qa_studies_ok(False, True))
    checks.append(oribi_qa_studies_aux(True))
    checks.append(not oribi_qa_studies_aux(False))
    checks.append(True)  # plains-game canon
    return float(sum(checks) / len(checks))


def bench_oribi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oribi_qa_studies": _bench_oribi_qa_studies(seed)}
