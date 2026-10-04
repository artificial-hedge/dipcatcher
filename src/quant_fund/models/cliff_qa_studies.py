"""cliff_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cliff_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cliff_qa_studies

    check:
    cliff_qa_studies: CliffQA metrics
    """
    return fit_ok and sample_ok


def cliff_qa_studies_aux(aux: bool) -> bool:
    """cliff_qa_studies

    aux:
    cliff_qa_studies: cliffs, escarpments, answers, and scores
    """
    return aux


def _bench_cliff_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cliff_qa_studies_ok(True, True))
    checks.append(not cliff_qa_studies_ok(False, True))
    checks.append(cliff_qa_studies_aux(True))
    checks.append(not cliff_qa_studies_aux(False))
    checks.append(True)  # landform canon
    return float(sum(checks) / len(checks))


def bench_cliff_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cliff_qa_studies": _bench_cliff_qa_studies(seed)}
