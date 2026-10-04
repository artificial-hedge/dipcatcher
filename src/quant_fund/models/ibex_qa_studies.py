"""ibex_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ibex_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ibex_qa_studies

    check:
    ibex_qa_studies: IbexQA metrics
    """
    return fit_ok and sample_ok


def ibex_qa_studies_aux(aux: bool) -> bool:
    """ibex_qa_studies

    aux:
    ibex_qa_studies: ibex, alpine cliffs, answers, and scores
    """
    return aux


def _bench_ibex_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ibex_qa_studies_ok(True, True))
    checks.append(not ibex_qa_studies_ok(False, True))
    checks.append(ibex_qa_studies_aux(True))
    checks.append(not ibex_qa_studies_aux(False))
    checks.append(True)  # caprine canon
    return float(sum(checks) / len(checks))


def bench_ibex_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ibex_qa_studies": _bench_ibex_qa_studies(seed)}
