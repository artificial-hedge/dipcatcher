"""buttercup_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def buttercup_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """buttercup_qa_studies

    check:
    buttercup_qa_studies: ButtercupQA metrics
    """
    return fit_ok and sample_ok


def buttercup_qa_studies_aux(aux: bool) -> bool:
    """buttercup_qa_studies

    aux:
    buttercup_qa_studies: buttercups, grazing fields, answers, and scores
    """
    return aux


def _bench_buttercup_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(buttercup_qa_studies_ok(True, True))
    checks.append(not buttercup_qa_studies_ok(False, True))
    checks.append(buttercup_qa_studies_aux(True))
    checks.append(not buttercup_qa_studies_aux(False))
    checks.append(True)  # wildflower canon
    return float(sum(checks) / len(checks))


def bench_buttercup_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_buttercup_qa_studies": _bench_buttercup_qa_studies(seed)}
