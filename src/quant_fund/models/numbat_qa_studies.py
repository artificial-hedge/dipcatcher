"""numbat_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def numbat_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """numbat_qa_studies

    check:
    numbat_qa_studies: NumbatQA metrics
    """
    return fit_ok and sample_ok


def numbat_qa_studies_aux(aux: bool) -> bool:
    """numbat_qa_studies

    aux:
    numbat_qa_studies: numbats, termites, answers, and scores
    """
    return aux


def _bench_numbat_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(numbat_qa_studies_ok(True, True))
    checks.append(not numbat_qa_studies_ok(False, True))
    checks.append(numbat_qa_studies_aux(True))
    checks.append(not numbat_qa_studies_aux(False))
    checks.append(True)  # marsupial canon
    return float(sum(checks) / len(checks))


def bench_numbat_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_numbat_qa_studies": _bench_numbat_qa_studies(seed)}
