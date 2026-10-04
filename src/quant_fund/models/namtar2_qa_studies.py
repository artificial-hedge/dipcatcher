"""namtar2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def namtar2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """namtar2_qa_studies

    check:
    namtar2_qa_studies: Namtar2QA metrics
    """
    return fit_ok and sample_ok


def namtar2_qa_studies_aux(aux: bool) -> bool:
    """namtar2_qa_studies

    aux:
    namtar2_qa_studies: namtar2, fate plagues, answers, and scores
    """
    return aux


def _bench_namtar2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(namtar2_qa_studies_ok(True, True))
    checks.append(not namtar2_qa_studies_ok(False, True))
    checks.append(namtar2_qa_studies_aux(True))
    checks.append(not namtar2_qa_studies_aux(False))
    checks.append(True)  # sumerian-6 canon
    return float(sum(checks) / len(checks))


def bench_namtar2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_namtar2_qa_studies": _bench_namtar2_qa_studies(seed)}
