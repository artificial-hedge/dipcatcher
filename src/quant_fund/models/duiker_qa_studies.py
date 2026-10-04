"""duiker_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def duiker_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """duiker_qa_studies

    check:
    duiker_qa_studies: DuikerQA metrics
    """
    return fit_ok and sample_ok


def duiker_qa_studies_aux(aux: bool) -> bool:
    """duiker_qa_studies

    aux:
    duiker_qa_studies: duikers, brush, answers, and scores
    """
    return aux


def _bench_duiker_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(duiker_qa_studies_ok(True, True))
    checks.append(not duiker_qa_studies_ok(False, True))
    checks.append(duiker_qa_studies_aux(True))
    checks.append(not duiker_qa_studies_aux(False))
    checks.append(True)  # antelope-2 canon
    return float(sum(checks) / len(checks))


def bench_duiker_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_duiker_qa_studies": _bench_duiker_qa_studies(seed)}
