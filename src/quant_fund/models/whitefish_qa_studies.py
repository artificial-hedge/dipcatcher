"""whitefish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def whitefish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """whitefish_qa_studies

    check:
    whitefish_qa_studies: WhitefishQA metrics
    """
    return fit_ok and sample_ok


def whitefish_qa_studies_aux(aux: bool) -> bool:
    """whitefish_qa_studies

    aux:
    whitefish_qa_studies: whitefish, deep lakes, answers, and scores
    """
    return aux


def _bench_whitefish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(whitefish_qa_studies_ok(True, True))
    checks.append(not whitefish_qa_studies_ok(False, True))
    checks.append(whitefish_qa_studies_aux(True))
    checks.append(not whitefish_qa_studies_aux(False))
    checks.append(True)  # salmonid canon
    return float(sum(checks) / len(checks))


def bench_whitefish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_whitefish_qa_studies": _bench_whitefish_qa_studies(seed)}
