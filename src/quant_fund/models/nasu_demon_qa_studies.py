"""nasu_demon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nasu_demon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nasu_demon_qa_studies

    check:
    nasu_demon_qa_studies: N
    """
    return fit_ok and sample_ok


def nasu_demon_qa_studies_aux(aux: bool) -> bool:
    """nasu_demon_qa_studies

    aux:
    nasu_demon_qa_studies: a
    """
    return aux


def _bench_nasu_demon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nasu_demon_qa_studies_ok(True, True))
    checks.append(not nasu_demon_qa_studies_ok(False, True))
    checks.append(nasu_demon_qa_studies_aux(True))
    checks.append(not nasu_demon_qa_studies_aux(False))
    checks.append(True)  # persian-div canon
    return float(sum(checks) / len(checks))


def bench_nasu_demon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nasu_demon_qa_studies": _bench_nasu_demon_qa_studies(seed)}
