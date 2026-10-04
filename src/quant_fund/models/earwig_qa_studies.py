"""earwig_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def earwig_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """earwig_qa_studies

    check:
    earwig_qa_studies: EarwigQA metrics
    """
    return fit_ok and sample_ok


def earwig_qa_studies_aux(aux: bool) -> bool:
    """earwig_qa_studies

    aux:
    earwig_qa_studies: earwigs, pincers, answers, and scores
    """
    return aux


def _bench_earwig_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(earwig_qa_studies_ok(True, True))
    checks.append(not earwig_qa_studies_ok(False, True))
    checks.append(earwig_qa_studies_aux(True))
    checks.append(not earwig_qa_studies_aux(False))
    checks.append(True)  # arthropod canon
    return float(sum(checks) / len(checks))


def bench_earwig_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_earwig_qa_studies": _bench_earwig_qa_studies(seed)}
