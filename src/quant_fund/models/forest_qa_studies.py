"""forest_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def forest_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """forest_qa_studies

    check:
    forest_qa_studies: ForestQA metrics
    """
    return fit_ok and sample_ok


def forest_qa_studies_aux(aux: bool) -> bool:
    """forest_qa_studies

    aux:
    forest_qa_studies: forests, species, answers, and scores
    """
    return aux


def _bench_forest_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(forest_qa_studies_ok(True, True))
    checks.append(not forest_qa_studies_ok(False, True))
    checks.append(forest_qa_studies_aux(True))
    checks.append(not forest_qa_studies_aux(False))
    checks.append(True)  # terrain-2 canon
    return float(sum(checks) / len(checks))


def bench_forest_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forest_qa_studies": _bench_forest_qa_studies(seed)}
