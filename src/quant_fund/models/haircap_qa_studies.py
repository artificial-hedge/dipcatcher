"""haircap_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def haircap_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """haircap_qa_studies

    check:
    haircap_qa_studies: HaircapQA metrics
    """
    return fit_ok and sample_ok


def haircap_qa_studies_aux(aux: bool) -> bool:
    """haircap_qa_studies

    aux:
    haircap_qa_studies: haircaps, heathlands, answers, and scores
    """
    return aux


def _bench_haircap_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(haircap_qa_studies_ok(True, True))
    checks.append(not haircap_qa_studies_ok(False, True))
    checks.append(haircap_qa_studies_aux(True))
    checks.append(not haircap_qa_studies_aux(False))
    checks.append(True)  # moss canon
    return float(sum(checks) / len(checks))


def bench_haircap_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_haircap_qa_studies": _bench_haircap_qa_studies(seed)}
