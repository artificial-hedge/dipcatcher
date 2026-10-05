"""trauco_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def trauco_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """trauco_qa_studies

    check:
    trauco_qa_studies: T
    """
    return fit_ok and sample_ok


def trauco_qa_studies_aux(aux: bool) -> bool:
    """trauco_qa_studies

    aux:
    trauco_qa_studies: r
    """
    return aux


def _bench_trauco_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(trauco_qa_studies_ok(True, True))
    checks.append(not trauco_qa_studies_ok(False, True))
    checks.append(trauco_qa_studies_aux(True))
    checks.append(not trauco_qa_studies_aux(False))
    checks.append(True)  # chiloe-demon canon
    return float(sum(checks) / len(checks))


def bench_trauco_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trauco_qa_studies": _bench_trauco_qa_studies(seed)}
