"""diti_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def diti_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """diti_qa_studies

    check:
    diti_qa_studies: D
    """
    return fit_ok and sample_ok


def diti_qa_studies_aux(aux: bool) -> bool:
    """diti_qa_studies

    aux:
    diti_qa_studies: i
    """
    return aux


def _bench_diti_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(diti_qa_studies_ok(True, True))
    checks.append(not diti_qa_studies_ok(False, True))
    checks.append(diti_qa_studies_aux(True))
    checks.append(not diti_qa_studies_aux(False))
    checks.append(True)  # hindu-demon canon
    return float(sum(checks) / len(checks))


def bench_diti_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diti_qa_studies": _bench_diti_qa_studies(seed)}
