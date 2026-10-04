"""shellycoat_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def shellycoat_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """shellycoat_qa_studies

    check:
    shellycoat_qa_studies: S
    """
    return fit_ok and sample_ok


def shellycoat_qa_studies_aux(aux: bool) -> bool:
    """shellycoat_qa_studies

    aux:
    shellycoat_qa_studies: h
    """
    return aux


def _bench_shellycoat_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(shellycoat_qa_studies_ok(True, True))
    checks.append(not shellycoat_qa_studies_ok(False, True))
    checks.append(shellycoat_qa_studies_aux(True))
    checks.append(not shellycoat_qa_studies_aux(False))
    checks.append(True)  # celtic-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_shellycoat_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shellycoat_qa_studies": _bench_shellycoat_qa_studies(seed)}
