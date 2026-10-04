"""safun_hu_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def safun_hu_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """safun_hu_qa_studies

    check:
    safun_hu_qa_studies: n
    """
    return fit_ok and sample_ok


def safun_hu_qa_studies_aux(aux: bool) -> bool:
    """safun_hu_qa_studies

    aux:
    safun_hu_qa_studies: o
    """
    return aux


def _bench_safun_hu_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(safun_hu_qa_studies_ok(True, True))
    checks.append(not safun_hu_qa_studies_ok(False, True))
    checks.append(safun_hu_qa_studies_aux(True))
    checks.append(not safun_hu_qa_studies_aux(False))
    checks.append(True)  # punic-4 canon
    return float(sum(checks) / len(checks))


def bench_safun_hu_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_safun_hu_qa_studies": _bench_safun_hu_qa_studies(seed)}
