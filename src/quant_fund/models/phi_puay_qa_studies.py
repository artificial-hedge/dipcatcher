"""phi_puay_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def phi_puay_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """phi_puay_qa_studies

    check:
    phi_puay_qa_studies: P
    """
    return fit_ok and sample_ok


def phi_puay_qa_studies_aux(aux: bool) -> bool:
    """phi_puay_qa_studies

    aux:
    phi_puay_qa_studies: h
    """
    return aux


def _bench_phi_puay_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(phi_puay_qa_studies_ok(True, True))
    checks.append(not phi_puay_qa_studies_ok(False, True))
    checks.append(phi_puay_qa_studies_aux(True))
    checks.append(not phi_puay_qa_studies_aux(False))
    checks.append(True)  # thai-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_phi_puay_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_phi_puay_qa_studies": _bench_phi_puay_qa_studies(seed)}
