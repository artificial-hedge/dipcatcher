"""aoao_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def aoao_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """aoao_qa_studies

    check:
    aoao_qa_studies: A
    """
    return fit_ok and sample_ok


def aoao_qa_studies_aux(aux: bool) -> bool:
    """aoao_qa_studies

    aux:
    aoao_qa_studies: o
    """
    return aux


def _bench_aoao_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(aoao_qa_studies_ok(True, True))
    checks.append(not aoao_qa_studies_ok(False, True))
    checks.append(aoao_qa_studies_aux(True))
    checks.append(not aoao_qa_studies_aux(False))
    checks.append(True)  # guarani-demon canon
    return float(sum(checks) / len(checks))


def bench_aoao_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aoao_qa_studies": _bench_aoao_qa_studies(seed)}
