"""glass_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def glass_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """glass_qa_studies

    check:
    glass_qa_studies: GlassQA metrics
    """
    return fit_ok and sample_ok


def glass_qa_studies_aux(aux: bool) -> bool:
    """glass_qa_studies

    aux:
    glass_qa_studies: glasses, tempering, answers, and scores
    """
    return aux


def _bench_glass_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(glass_qa_studies_ok(True, True))
    checks.append(not glass_qa_studies_ok(False, True))
    checks.append(glass_qa_studies_aux(True))
    checks.append(not glass_qa_studies_aux(False))
    checks.append(True)  # material canon
    return float(sum(checks) / len(checks))


def bench_glass_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_glass_qa_studies": _bench_glass_qa_studies(seed)}
