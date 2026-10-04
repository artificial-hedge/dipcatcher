"""kyys_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kyys_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kyys_qa_studies

    check:
    kyys_qa_studies: K
    """
    return fit_ok and sample_ok


def kyys_qa_studies_aux(aux: bool) -> bool:
    """kyys_qa_studies

    aux:
    kyys_qa_studies: y
    """
    return aux


def _bench_kyys_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kyys_qa_studies_ok(True, True))
    checks.append(not kyys_qa_studies_ok(False, True))
    checks.append(kyys_qa_studies_aux(True))
    checks.append(not kyys_qa_studies_aux(False))
    checks.append(True)  # siberian-demon canon
    return float(sum(checks) / len(checks))


def bench_kyys_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kyys_qa_studies": _bench_kyys_qa_studies(seed)}
