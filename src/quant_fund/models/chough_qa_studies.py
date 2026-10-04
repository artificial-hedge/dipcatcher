"""chough_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def chough_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """chough_qa_studies

    check:
    chough_qa_studies: ChoughQA metrics
    """
    return fit_ok and sample_ok


def chough_qa_studies_aux(aux: bool) -> bool:
    """chough_qa_studies

    aux:
    chough_qa_studies: choughs, cliffs, answers, and scores
    """
    return aux


def _bench_chough_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(chough_qa_studies_ok(True, True))
    checks.append(not chough_qa_studies_ok(False, True))
    checks.append(chough_qa_studies_aux(True))
    checks.append(not chough_qa_studies_aux(False))
    checks.append(True)  # corvid canon
    return float(sum(checks) / len(checks))


def bench_chough_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chough_qa_studies": _bench_chough_qa_studies(seed)}
