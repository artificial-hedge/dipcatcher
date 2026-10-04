"""calcite_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def calcite_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """calcite_qa_studies

    check:
    calcite_qa_studies: CalciteQA metrics
    """
    return fit_ok and sample_ok


def calcite_qa_studies_aux(aux: bool) -> bool:
    """calcite_qa_studies

    aux:
    calcite_qa_studies: calcites, limestones, answers, and scores
    """
    return aux


def _bench_calcite_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(calcite_qa_studies_ok(True, True))
    checks.append(not calcite_qa_studies_ok(False, True))
    checks.append(calcite_qa_studies_aux(True))
    checks.append(not calcite_qa_studies_aux(False))
    checks.append(True)  # mineral canon
    return float(sum(checks) / len(checks))


def bench_calcite_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_calcite_qa_studies": _bench_calcite_qa_studies(seed)}
