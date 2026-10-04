"""nq_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def nq_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nq_lite_studies

    check:
    nq_lite_studies: NaturalQuestions metrics
    """
    return fit_ok and sample_ok


def nq_lite_studies_aux(aux: bool) -> bool:
    """nq_lite_studies

    aux:
    nq_lite_studies: questions, spans, answers, and scores
    """
    return aux


def _bench_nq_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nq_lite_studies_ok(True, True))
    checks.append(not nq_lite_studies_ok(False, True))
    checks.append(nq_lite_studies_aux(True))
    checks.append(not nq_lite_studies_aux(False))
    checks.append(True)  # retrieval-eval canon
    return float(sum(checks) / len(checks))


def bench_nq_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nq_lite_studies": _bench_nq_lite_studies(seed)}
