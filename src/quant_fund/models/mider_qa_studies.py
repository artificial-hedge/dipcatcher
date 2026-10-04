"""mider_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mider_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mider_qa_studies

    check:
    mider_qa_studies: d
    """
    return fit_ok and sample_ok


def mider_qa_studies_aux(aux: bool) -> bool:
    """mider_qa_studies

    aux:
    mider_qa_studies: i
    """
    return aux


def _bench_mider_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mider_qa_studies_ok(True, True))
    checks.append(not mider_qa_studies_ok(False, True))
    checks.append(mider_qa_studies_aux(True))
    checks.append(not mider_qa_studies_aux(False))
    checks.append(True)  # punic-3 canon
    return float(sum(checks) / len(checks))


def bench_mider_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mider_qa_studies": _bench_mider_qa_studies(seed)}
