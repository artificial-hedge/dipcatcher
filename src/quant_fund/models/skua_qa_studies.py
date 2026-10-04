"""skua_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def skua_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """skua_qa_studies

    check:
    skua_qa_studies: SkuaQA metrics
    """
    return fit_ok and sample_ok


def skua_qa_studies_aux(aux: bool) -> bool:
    """skua_qa_studies

    aux:
    skua_qa_studies: skuas, raids, answers, and scores
    """
    return aux


def _bench_skua_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(skua_qa_studies_ok(True, True))
    checks.append(not skua_qa_studies_ok(False, True))
    checks.append(skua_qa_studies_aux(True))
    checks.append(not skua_qa_studies_aux(False))
    checks.append(True)  # seabird canon
    return float(sum(checks) / len(checks))


def bench_skua_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_skua_qa_studies": _bench_skua_qa_studies(seed)}
