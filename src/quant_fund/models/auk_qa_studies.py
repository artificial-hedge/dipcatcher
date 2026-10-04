"""auk_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def auk_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """auk_qa_studies

    check:
    auk_qa_studies: AukQA metrics
    """
    return fit_ok and sample_ok


def auk_qa_studies_aux(aux: bool) -> bool:
    """auk_qa_studies

    aux:
    auk_qa_studies: auks, ledges, answers, and scores
    """
    return aux


def _bench_auk_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(auk_qa_studies_ok(True, True))
    checks.append(not auk_qa_studies_ok(False, True))
    checks.append(auk_qa_studies_aux(True))
    checks.append(not auk_qa_studies_aux(False))
    checks.append(True)  # seabird-2 canon
    return float(sum(checks) / len(checks))


def bench_auk_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_auk_qa_studies": _bench_auk_qa_studies(seed)}
