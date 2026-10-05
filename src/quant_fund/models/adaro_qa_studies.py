"""adaro_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def adaro_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """adaro_qa_studies

    check:
    adaro_qa_studies: A
    """
    return fit_ok and sample_ok


def adaro_qa_studies_aux(aux: bool) -> bool:
    """adaro_qa_studies

    aux:
    adaro_qa_studies: d
    """
    return aux


def _bench_adaro_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(adaro_qa_studies_ok(True, True))
    checks.append(not adaro_qa_studies_ok(False, True))
    checks.append(adaro_qa_studies_aux(True))
    checks.append(not adaro_qa_studies_aux(False))
    checks.append(True)  # oceania-demon canon
    return float(sum(checks) / len(checks))


def bench_adaro_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adaro_qa_studies": _bench_adaro_qa_studies(seed)}
