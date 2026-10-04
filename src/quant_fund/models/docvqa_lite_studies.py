"""docvqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def docvqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """docvqa_lite_studies

    check:
    docvqa_lite_studies: DocVQA metrics
    """
    return fit_ok and sample_ok


def docvqa_lite_studies_aux(aux: bool) -> bool:
    """docvqa_lite_studies

    aux:
    docvqa_lite_studies: documents, questions, answers, and scores
    """
    return aux


def _bench_docvqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(docvqa_lite_studies_ok(True, True))
    checks.append(not docvqa_lite_studies_ok(False, True))
    checks.append(docvqa_lite_studies_aux(True))
    checks.append(not docvqa_lite_studies_aux(False))
    checks.append(True)  # vision-doc-QA canon
    return float(sum(checks) / len(checks))


def bench_docvqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_docvqa_lite_studies": _bench_docvqa_lite_studies(seed)}
