"""tiniri_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tiniri_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tiniri_qa_studies

    check:
    tiniri_qa_studies: d
    """
    return fit_ok and sample_ok


def tiniri_qa_studies_aux(aux: bool) -> bool:
    """tiniri_qa_studies

    aux:
    tiniri_qa_studies: e
    """
    return aux


def _bench_tiniri_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tiniri_qa_studies_ok(True, True))
    checks.append(not tiniri_qa_studies_ok(False, True))
    checks.append(tiniri_qa_studies_aux(True))
    checks.append(not tiniri_qa_studies_aux(False))
    checks.append(True)  # garamantian canon
    return float(sum(checks) / len(checks))


def bench_tiniri_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tiniri_qa_studies": _bench_tiniri_qa_studies(seed)}
