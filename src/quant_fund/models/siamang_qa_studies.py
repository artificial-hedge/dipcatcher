"""siamang_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def siamang_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """siamang_qa_studies

    check:
    siamang_qa_studies: SiamangQA metrics
    """
    return fit_ok and sample_ok


def siamang_qa_studies_aux(aux: bool) -> bool:
    """siamang_qa_studies

    aux:
    siamang_qa_studies: siamangs, sumatran canopy, answers, and scores
    """
    return aux


def _bench_siamang_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(siamang_qa_studies_ok(True, True))
    checks.append(not siamang_qa_studies_ok(False, True))
    checks.append(siamang_qa_studies_aux(True))
    checks.append(not siamang_qa_studies_aux(False))
    checks.append(True)  # primate-2 canon
    return float(sum(checks) / len(checks))


def bench_siamang_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_siamang_qa_studies": _bench_siamang_qa_studies(seed)}
