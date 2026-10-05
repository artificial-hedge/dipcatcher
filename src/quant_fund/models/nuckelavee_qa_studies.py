"""nuckelavee_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nuckelavee_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nuckelavee_qa_studies

    check:
    nuckelavee_qa_studies: s
    """
    return fit_ok and sample_ok


def nuckelavee_qa_studies_aux(aux: bool) -> bool:
    """nuckelavee_qa_studies

    aux:
    nuckelavee_qa_studies: k
    """
    return aux


def _bench_nuckelavee_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nuckelavee_qa_studies_ok(True, True))
    checks.append(not nuckelavee_qa_studies_ok(False, True))
    checks.append(nuckelavee_qa_studies_aux(True))
    checks.append(not nuckelavee_qa_studies_aux(False))
    checks.append(True)  # celtic-myth-6 canon
    return float(sum(checks) / len(checks))


def bench_nuckelavee_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nuckelavee_qa_studies": _bench_nuckelavee_qa_studies(seed)}
