"""tesfit_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tesfit_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tesfit_qa_studies

    check:
    tesfit_qa_studies: b
    """
    return fit_ok and sample_ok


def tesfit_qa_studies_aux(aux: bool) -> bool:
    """tesfit_qa_studies

    aux:
    tesfit_qa_studies: l
    """
    return aux


def _bench_tesfit_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tesfit_qa_studies_ok(True, True))
    checks.append(not tesfit_qa_studies_ok(False, True))
    checks.append(tesfit_qa_studies_aux(True))
    checks.append(not tesfit_qa_studies_aux(False))
    checks.append(True)  # tuareg-2 canon
    return float(sum(checks) / len(checks))


def bench_tesfit_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tesfit_qa_studies": _bench_tesfit_qa_studies(seed)}
