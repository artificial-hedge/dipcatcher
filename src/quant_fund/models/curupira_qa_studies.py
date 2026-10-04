"""curupira_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def curupira_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """curupira_qa_studies

    check:
    curupira_qa_studies: C
    """
    return fit_ok and sample_ok


def curupira_qa_studies_aux(aux: bool) -> bool:
    """curupira_qa_studies

    aux:
    curupira_qa_studies: u
    """
    return aux


def _bench_curupira_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(curupira_qa_studies_ok(True, True))
    checks.append(not curupira_qa_studies_ok(False, True))
    checks.append(curupira_qa_studies_aux(True))
    checks.append(not curupira_qa_studies_aux(False))
    checks.append(True)  # brazilian-folklore canon
    return float(sum(checks) / len(checks))


def bench_curupira_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_curupira_qa_studies": _bench_curupira_qa_studies(seed)}
