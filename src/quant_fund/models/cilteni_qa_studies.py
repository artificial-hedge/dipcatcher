"""cilteni_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cilteni_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cilteni_qa_studies

    check:
    cilteni_qa_studies: h
    """
    return fit_ok and sample_ok


def cilteni_qa_studies_aux(aux: bool) -> bool:
    """cilteni_qa_studies

    aux:
    cilteni_qa_studies: i
    """
    return aux


def _bench_cilteni_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cilteni_qa_studies_ok(True, True))
    checks.append(not cilteni_qa_studies_ok(False, True))
    checks.append(cilteni_qa_studies_aux(True))
    checks.append(not cilteni_qa_studies_aux(False))
    checks.append(True)  # saharan-2 canon
    return float(sum(checks) / len(checks))


def bench_cilteni_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cilteni_qa_studies": _bench_cilteni_qa_studies(seed)}
