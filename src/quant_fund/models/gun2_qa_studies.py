"""gun2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gun2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gun2_qa_studies

    check:
    gun2_qa_studies: Gun2QA metrics
    """
    return fit_ok and sample_ok


def gun2_qa_studies_aux(aux: bool) -> bool:
    """gun2_qa_studies

    aux:
    gun2_qa_studies: gun2, flood builders, answers, and scores
    """
    return aux


def _bench_gun2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gun2_qa_studies_ok(True, True))
    checks.append(not gun2_qa_studies_ok(False, True))
    checks.append(gun2_qa_studies_aux(True))
    checks.append(not gun2_qa_studies_aux(False))
    checks.append(True)  # chinese-myth-7 canon
    return float(sum(checks) / len(checks))


def bench_gun2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gun2_qa_studies": _bench_gun2_qa_studies(seed)}
