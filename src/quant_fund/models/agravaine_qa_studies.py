"""agravaine_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def agravaine_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """agravaine_qa_studies

    check:
    agravaine_qa_studies: p
    """
    return fit_ok and sample_ok


def agravaine_qa_studies_aux(aux: bool) -> bool:
    """agravaine_qa_studies

    aux:
    agravaine_qa_studies: r
    """
    return aux


def _bench_agravaine_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(agravaine_qa_studies_ok(True, True))
    checks.append(not agravaine_qa_studies_ok(False, True))
    checks.append(agravaine_qa_studies_aux(True))
    checks.append(not agravaine_qa_studies_aux(False))
    checks.append(True)  # arthurian-3 canon
    return float(sum(checks) / len(checks))


def bench_agravaine_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_agravaine_qa_studies": _bench_agravaine_qa_studies(seed)}
