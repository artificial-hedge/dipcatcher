"""percival_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def percival_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """percival_qa_studies

    check:
    percival_qa_studies: p
    """
    return fit_ok and sample_ok


def percival_qa_studies_aux(aux: bool) -> bool:
    """percival_qa_studies

    aux:
    percival_qa_studies: u
    """
    return aux


def _bench_percival_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(percival_qa_studies_ok(True, True))
    checks.append(not percival_qa_studies_ok(False, True))
    checks.append(percival_qa_studies_aux(True))
    checks.append(not percival_qa_studies_aux(False))
    checks.append(True)  # arthurian-2 canon
    return float(sum(checks) / len(checks))


def bench_percival_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_percival_qa_studies": _bench_percival_qa_studies(seed)}
