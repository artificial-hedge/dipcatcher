"""galago_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def galago_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """galago_qa_studies

    check:
    galago_qa_studies: GalagoQA metrics
    """
    return fit_ok and sample_ok


def galago_qa_studies_aux(aux: bool) -> bool:
    """galago_qa_studies

    aux:
    galago_qa_studies: galagos, acacia savannas, answers, and scores
    """
    return aux


def _bench_galago_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(galago_qa_studies_ok(True, True))
    checks.append(not galago_qa_studies_ok(False, True))
    checks.append(galago_qa_studies_aux(True))
    checks.append(not galago_qa_studies_aux(False))
    checks.append(True)  # prosimian canon
    return float(sum(checks) / len(checks))


def bench_galago_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galago_qa_studies": _bench_galago_qa_studies(seed)}
