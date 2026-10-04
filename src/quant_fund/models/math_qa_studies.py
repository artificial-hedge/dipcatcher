"""math_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def math_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """math_qa_studies

    check:
    math_qa_studies: MathQA metrics
    """
    return fit_ok and sample_ok


def math_qa_studies_aux(aux: bool) -> bool:
    """math_qa_studies

    aux:
    math_qa_studies: problems, programs, answers, and scores
    """
    return aux


def _bench_math_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(math_qa_studies_ok(True, True))
    checks.append(not math_qa_studies_ok(False, True))
    checks.append(math_qa_studies_aux(True))
    checks.append(not math_qa_studies_aux(False))
    checks.append(True)  # numerical-reasoning canon
    return float(sum(checks) / len(checks))


def bench_math_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_math_qa_studies": _bench_math_qa_studies(seed)}
