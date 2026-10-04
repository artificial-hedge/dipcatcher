"""imajeghen_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def imajeghen_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """imajeghen_qa_studies

    check:
    imajeghen_qa_studies: n
    """
    return fit_ok and sample_ok


def imajeghen_qa_studies_aux(aux: bool) -> bool:
    """imajeghen_qa_studies

    aux:
    imajeghen_qa_studies: o
    """
    return aux


def _bench_imajeghen_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(imajeghen_qa_studies_ok(True, True))
    checks.append(not imajeghen_qa_studies_ok(False, True))
    checks.append(imajeghen_qa_studies_aux(True))
    checks.append(not imajeghen_qa_studies_aux(False))
    checks.append(True)  # tuareg-3 canon
    return float(sum(checks) / len(checks))


def bench_imajeghen_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_imajeghen_qa_studies": _bench_imajeghen_qa_studies(seed)}
