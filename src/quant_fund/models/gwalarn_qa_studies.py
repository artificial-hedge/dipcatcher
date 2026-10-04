"""gwalarn_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gwalarn_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gwalarn_qa_studies

    check:
    gwalarn_qa_studies: n
    """
    return fit_ok and sample_ok


def gwalarn_qa_studies_aux(aux: bool) -> bool:
    """gwalarn_qa_studies

    aux:
    gwalarn_qa_studies: o
    """
    return aux


def _bench_gwalarn_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gwalarn_qa_studies_ok(True, True))
    checks.append(not gwalarn_qa_studies_ok(False, True))
    checks.append(gwalarn_qa_studies_aux(True))
    checks.append(not gwalarn_qa_studies_aux(False))
    checks.append(True)  # breton-myth canon
    return float(sum(checks) / len(checks))


def bench_gwalarn_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gwalarn_qa_studies": _bench_gwalarn_qa_studies(seed)}
