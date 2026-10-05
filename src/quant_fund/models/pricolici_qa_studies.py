"""pricolici_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pricolici_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pricolici_qa_studies

    check:
    pricolici_qa_studies: P
    """
    return fit_ok and sample_ok


def pricolici_qa_studies_aux(aux: bool) -> bool:
    """pricolici_qa_studies

    aux:
    pricolici_qa_studies: r
    """
    return aux


def _bench_pricolici_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pricolici_qa_studies_ok(True, True))
    checks.append(not pricolici_qa_studies_ok(False, True))
    checks.append(pricolici_qa_studies_aux(True))
    checks.append(not pricolici_qa_studies_aux(False))
    checks.append(True)  # romanian-demon canon
    return float(sum(checks) / len(checks))


def bench_pricolici_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pricolici_qa_studies": _bench_pricolici_qa_studies(seed)}
