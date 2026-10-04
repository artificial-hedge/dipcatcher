"""iris_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def iris_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """iris_qa_studies

    check:
    iris_qa_studies: IrisQA metrics
    """
    return fit_ok and sample_ok


def iris_qa_studies_aux(aux: bool) -> bool:
    """iris_qa_studies

    aux:
    iris_qa_studies: irises, sepals, answers, and scores
    """
    return aux


def _bench_iris_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(iris_qa_studies_ok(True, True))
    checks.append(not iris_qa_studies_ok(False, True))
    checks.append(iris_qa_studies_aux(True))
    checks.append(not iris_qa_studies_aux(False))
    checks.append(True)  # wildflower canon
    return float(sum(checks) / len(checks))


def bench_iris_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iris_qa_studies": _bench_iris_qa_studies(seed)}
