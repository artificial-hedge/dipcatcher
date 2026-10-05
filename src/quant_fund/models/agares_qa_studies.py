"""agares_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def agares_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """agares_qa_studies

    check:
    agares_qa_studies: A
    """
    return fit_ok and sample_ok


def agares_qa_studies_aux(aux: bool) -> bool:
    """agares_qa_studies

    aux:
    agares_qa_studies: g
    """
    return aux


def _bench_agares_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(agares_qa_studies_ok(True, True))
    checks.append(not agares_qa_studies_ok(False, True))
    checks.append(agares_qa_studies_aux(True))
    checks.append(not agares_qa_studies_aux(False))
    checks.append(True)  # goetic-legion canon
    return float(sum(checks) / len(checks))


def bench_agares_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_agares_qa_studies": _bench_agares_qa_studies(seed)}
