"""mayfly_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mayfly_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mayfly_qa_studies

    check:
    mayfly_qa_studies: MayflyQA metrics
    """
    return fit_ok and sample_ok


def mayfly_qa_studies_aux(aux: bool) -> bool:
    """mayfly_qa_studies

    aux:
    mayfly_qa_studies: mayflies, emergences, answers, and scores
    """
    return aux


def _bench_mayfly_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mayfly_qa_studies_ok(True, True))
    checks.append(not mayfly_qa_studies_ok(False, True))
    checks.append(mayfly_qa_studies_aux(True))
    checks.append(not mayfly_qa_studies_aux(False))
    checks.append(True)  # arthropod canon
    return float(sum(checks) / len(checks))


def bench_mayfly_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mayfly_qa_studies": _bench_mayfly_qa_studies(seed)}
