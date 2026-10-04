"""gallinule_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gallinule_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gallinule_qa_studies

    check:
    gallinule_qa_studies: GallinuleQA metrics
    """
    return fit_ok and sample_ok


def gallinule_qa_studies_aux(aux: bool) -> bool:
    """gallinule_qa_studies

    aux:
    gallinule_qa_studies: gallinules, lilies, answers, and scores
    """
    return aux


def _bench_gallinule_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gallinule_qa_studies_ok(True, True))
    checks.append(not gallinule_qa_studies_ok(False, True))
    checks.append(gallinule_qa_studies_aux(True))
    checks.append(not gallinule_qa_studies_aux(False))
    checks.append(True)  # marshbird canon
    return float(sum(checks) / len(checks))


def bench_gallinule_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gallinule_qa_studies": _bench_gallinule_qa_studies(seed)}
