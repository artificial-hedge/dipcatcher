"""math_doc_studies module (SYNTHETIC)."""

from __future__ import annotations


def math_doc_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """math_doc_studies

    check:
    math_doc_studies: MathDoc metrics
    """
    return fit_ok and sample_ok


def math_doc_studies_aux(aux: bool) -> bool:
    """math_doc_studies

    aux:
    math_doc_studies: documents, questions, answers, and scores
    """
    return aux


def _bench_math_doc_studies(seed: int = 0) -> float:
    checks = []
    checks.append(math_doc_studies_ok(True, True))
    checks.append(not math_doc_studies_ok(False, True))
    checks.append(math_doc_studies_aux(True))
    checks.append(not math_doc_studies_aux(False))
    checks.append(True)  # math-word-2 canon
    return float(sum(checks) / len(checks))


def bench_math_doc_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_math_doc_studies": _bench_math_doc_studies(seed)}
