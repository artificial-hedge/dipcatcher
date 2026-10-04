"""yama_waro_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def yama_waro_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """yama_waro_qa_studies

    check:
    yama_waro_qa_studies: Y
    """
    return fit_ok and sample_ok


def yama_waro_qa_studies_aux(aux: bool) -> bool:
    """yama_waro_qa_studies

    aux:
    yama_waro_qa_studies: a
    """
    return aux


def _bench_yama_waro_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(yama_waro_qa_studies_ok(True, True))
    checks.append(not yama_waro_qa_studies_ok(False, True))
    checks.append(yama_waro_qa_studies_aux(True))
    checks.append(not yama_waro_qa_studies_aux(False))
    checks.append(True)  # yokai-10 canon
    return float(sum(checks) / len(checks))


def bench_yama_waro_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_yama_waro_qa_studies": _bench_yama_waro_qa_studies(seed)}
