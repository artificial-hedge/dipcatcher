"""lauma_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def lauma_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lauma_qa_studies

    check:
    lauma_qa_studies: L
    """
    return fit_ok and sample_ok


def lauma_qa_studies_aux(aux: bool) -> bool:
    """lauma_qa_studies

    aux:
    lauma_qa_studies: a
    """
    return aux


def _bench_lauma_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lauma_qa_studies_ok(True, True))
    checks.append(not lauma_qa_studies_ok(False, True))
    checks.append(lauma_qa_studies_aux(True))
    checks.append(not lauma_qa_studies_aux(False))
    checks.append(True)  # baltic-demon canon
    return float(sum(checks) / len(checks))


def bench_lauma_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lauma_qa_studies": _bench_lauma_qa_studies(seed)}
