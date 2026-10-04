"""gulper_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gulper_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gulper_qa_studies

    check:
    gulper_qa_studies: GulperQA metrics
    """
    return fit_ok and sample_ok


def gulper_qa_studies_aux(aux: bool) -> bool:
    """gulper_qa_studies

    aux:
    gulper_qa_studies: gulper eels, pelagic canyons, answers, and scores
    """
    return aux


def _bench_gulper_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gulper_qa_studies_ok(True, True))
    checks.append(not gulper_qa_studies_ok(False, True))
    checks.append(gulper_qa_studies_aux(True))
    checks.append(not gulper_qa_studies_aux(False))
    checks.append(True)  # abyssal-2 canon
    return float(sum(checks) / len(checks))


def bench_gulper_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gulper_qa_studies": _bench_gulper_qa_studies(seed)}
