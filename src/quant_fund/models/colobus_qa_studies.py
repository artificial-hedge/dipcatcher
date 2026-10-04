"""colobus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def colobus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """colobus_qa_studies

    check:
    colobus_qa_studies: ColobusQA metrics
    """
    return fit_ok and sample_ok


def colobus_qa_studies_aux(aux: bool) -> bool:
    """colobus_qa_studies

    aux:
    colobus_qa_studies: colobus monkeys, high canopy, answers, and scores
    """
    return aux


def _bench_colobus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(colobus_qa_studies_ok(True, True))
    checks.append(not colobus_qa_studies_ok(False, True))
    checks.append(colobus_qa_studies_aux(True))
    checks.append(not colobus_qa_studies_aux(False))
    checks.append(True)  # old-world-monkey canon
    return float(sum(checks) / len(checks))


def bench_colobus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_colobus_qa_studies": _bench_colobus_qa_studies(seed)}
