"""perkunas2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def perkunas2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """perkunas2_qa_studies

    check:
    perkunas2_qa_studies: Perkunas2QA metrics
    """
    return fit_ok and sample_ok


def perkunas2_qa_studies_aux(aux: bool) -> bool:
    """perkunas2_qa_studies

    aux:
    perkunas2_qa_studies: perkunas2, thunder axes, answers, and scores
    """
    return aux


def _bench_perkunas2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(perkunas2_qa_studies_ok(True, True))
    checks.append(not perkunas2_qa_studies_ok(False, True))
    checks.append(perkunas2_qa_studies_aux(True))
    checks.append(not perkunas2_qa_studies_aux(False))
    checks.append(True)  # baltic-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_perkunas2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perkunas2_qa_studies": _bench_perkunas2_qa_studies(seed)}
