"""uther_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def uther_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """uther_qa_studies

    check:
    uther_qa_studies: p
    """
    return fit_ok and sample_ok


def uther_qa_studies_aux(aux: bool) -> bool:
    """uther_qa_studies

    aux:
    uther_qa_studies: e
    """
    return aux


def _bench_uther_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(uther_qa_studies_ok(True, True))
    checks.append(not uther_qa_studies_ok(False, True))
    checks.append(uther_qa_studies_aux(True))
    checks.append(not uther_qa_studies_aux(False))
    checks.append(True)  # arthurian-4 canon
    return float(sum(checks) / len(checks))


def bench_uther_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uther_qa_studies": _bench_uther_qa_studies(seed)}
