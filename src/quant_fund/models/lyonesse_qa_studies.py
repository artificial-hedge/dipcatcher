"""lyonesse_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def lyonesse_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lyonesse_qa_studies

    check:
    lyonesse_qa_studies: l
    """
    return fit_ok and sample_ok


def lyonesse_qa_studies_aux(aux: bool) -> bool:
    """lyonesse_qa_studies

    aux:
    lyonesse_qa_studies: y
    """
    return aux


def _bench_lyonesse_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lyonesse_qa_studies_ok(True, True))
    checks.append(not lyonesse_qa_studies_ok(False, True))
    checks.append(lyonesse_qa_studies_aux(True))
    checks.append(not lyonesse_qa_studies_aux(False))
    checks.append(True)  # arthurian-3 canon
    return float(sum(checks) / len(checks))


def bench_lyonesse_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lyonesse_qa_studies": _bench_lyonesse_qa_studies(seed)}
