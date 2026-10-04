"""carex_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def carex_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """carex_qa_studies

    check:
    carex_qa_studies: CarexQA metrics
    """
    return fit_ok and sample_ok


def carex_qa_studies_aux(aux: bool) -> bool:
    """carex_qa_studies

    aux:
    carex_qa_studies: carexes, fens, answers, and scores
    """
    return aux


def _bench_carex_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(carex_qa_studies_ok(True, True))
    checks.append(not carex_qa_studies_ok(False, True))
    checks.append(carex_qa_studies_aux(True))
    checks.append(not carex_qa_studies_aux(False))
    checks.append(True)  # sedge canon
    return float(sum(checks) / len(checks))


def bench_carex_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_carex_qa_studies": _bench_carex_qa_studies(seed)}
