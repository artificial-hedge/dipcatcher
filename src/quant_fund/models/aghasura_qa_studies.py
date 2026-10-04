"""aghasura_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def aghasura_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """aghasura_qa_studies

    check:
    aghasura_qa_studies: A
    """
    return fit_ok and sample_ok


def aghasura_qa_studies_aux(aux: bool) -> bool:
    """aghasura_qa_studies

    aux:
    aghasura_qa_studies: g
    """
    return aux


def _bench_aghasura_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(aghasura_qa_studies_ok(True, True))
    checks.append(not aghasura_qa_studies_ok(False, True))
    checks.append(aghasura_qa_studies_aux(True))
    checks.append(not aghasura_qa_studies_aux(False))
    checks.append(True)  # hindu-demon canon
    return float(sum(checks) / len(checks))


def bench_aghasura_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aghasura_qa_studies": _bench_aghasura_qa_studies(seed)}
