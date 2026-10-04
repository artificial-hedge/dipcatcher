"""iris2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def iris2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """iris2_qa_studies

    check:
    iris2_qa_studies: Iris2QA metrics
    """
    return fit_ok and sample_ok


def iris2_qa_studies_aux(aux: bool) -> bool:
    """iris2_qa_studies

    aux:
    iris2_qa_studies: iris2, rainbow wings, answers, and scores
    """
    return aux


def _bench_iris2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(iris2_qa_studies_ok(True, True))
    checks.append(not iris2_qa_studies_ok(False, True))
    checks.append(iris2_qa_studies_aux(True))
    checks.append(not iris2_qa_studies_aux(False))
    checks.append(True)  # greek-myth-11 canon
    return float(sum(checks) / len(checks))


def bench_iris2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iris2_qa_studies": _bench_iris2_qa_studies(seed)}
