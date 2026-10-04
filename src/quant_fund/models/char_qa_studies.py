"""char_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def char_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """char_qa_studies

    check:
    char_qa_studies: CharQA metrics
    """
    return fit_ok and sample_ok


def char_qa_studies_aux(aux: bool) -> bool:
    """char_qa_studies

    aux:
    char_qa_studies: char, arctic lakes, answers, and scores
    """
    return aux


def _bench_char_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(char_qa_studies_ok(True, True))
    checks.append(not char_qa_studies_ok(False, True))
    checks.append(char_qa_studies_aux(True))
    checks.append(not char_qa_studies_aux(False))
    checks.append(True)  # salmonid canon
    return float(sum(checks) / len(checks))


def bench_char_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_char_qa_studies": _bench_char_qa_studies(seed)}
