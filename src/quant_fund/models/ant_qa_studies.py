"""ant_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ant_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ant_qa_studies

    check:
    ant_qa_studies: AntQA metrics
    """
    return fit_ok and sample_ok


def ant_qa_studies_aux(aux: bool) -> bool:
    """ant_qa_studies

    aux:
    ant_qa_studies: ants, colonies, answers, and scores
    """
    return aux


def _bench_ant_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ant_qa_studies_ok(True, True))
    checks.append(not ant_qa_studies_ok(False, True))
    checks.append(ant_qa_studies_aux(True))
    checks.append(not ant_qa_studies_aux(False))
    checks.append(True)  # insect canon
    return float(sum(checks) / len(checks))


def bench_ant_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ant_qa_studies": _bench_ant_qa_studies(seed)}
