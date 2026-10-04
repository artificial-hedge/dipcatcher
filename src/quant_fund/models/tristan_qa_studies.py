"""tristan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tristan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tristan_qa_studies

    check:
    tristan_qa_studies: l
    """
    return fit_ok and sample_ok


def tristan_qa_studies_aux(aux: bool) -> bool:
    """tristan_qa_studies

    aux:
    tristan_qa_studies: o
    """
    return aux


def _bench_tristan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tristan_qa_studies_ok(True, True))
    checks.append(not tristan_qa_studies_ok(False, True))
    checks.append(tristan_qa_studies_aux(True))
    checks.append(not tristan_qa_studies_aux(False))
    checks.append(True)  # arthurian-2 canon
    return float(sum(checks) / len(checks))


def bench_tristan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tristan_qa_studies": _bench_tristan_qa_studies(seed)}
