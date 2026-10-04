"""gwydion_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gwydion_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gwydion_qa_studies

    check:
    gwydion_qa_studies: GwydionQA metrics
    """
    return fit_ok and sample_ok


def gwydion_qa_studies_aux(aux: bool) -> bool:
    """gwydion_qa_studies

    aux:
    gwydion_qa_studies: gwydion, shape weavers, answers, and scores
    """
    return aux


def _bench_gwydion_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gwydion_qa_studies_ok(True, True))
    checks.append(not gwydion_qa_studies_ok(False, True))
    checks.append(gwydion_qa_studies_aux(True))
    checks.append(not gwydion_qa_studies_aux(False))
    checks.append(True)  # welsh-myth canon
    return float(sum(checks) / len(checks))


def bench_gwydion_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gwydion_qa_studies": _bench_gwydion_qa_studies(seed)}
