"""openbookqa_studies module (SYNTHETIC)."""

from __future__ import annotations


def openbookqa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """openbookqa_studies

    check:
    openbookqa_studies: OpenBookQA science facts and multiple-choice
    """
    return fit_ok and sample_ok


def openbookqa_studies_aux(aux: bool) -> bool:
    """openbookqa_studies

    aux:
    openbookqa_studies: question stems, fact lookups, and accuracy
    """
    return aux


def _bench_openbookqa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(openbookqa_studies_ok(True, True))
    checks.append(not openbookqa_studies_ok(False, True))
    checks.append(openbookqa_studies_aux(True))
    checks.append(not openbookqa_studies_aux(False))
    checks.append(True)  # commonsense-eval canon
    return float(sum(checks) / len(checks))


def bench_openbookqa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_openbookqa_studies": _bench_openbookqa_studies(seed)}
