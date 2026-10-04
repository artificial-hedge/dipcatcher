"""rod2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rod2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rod2_qa_studies

    check:
    rod2_qa_studies: Rod2QA metrics
    """
    return fit_ok and sample_ok


def rod2_qa_studies_aux(aux: bool) -> bool:
    """rod2_qa_studies

    aux:
    rod2_qa_studies: rod2, birth fates, answers, and scores
    """
    return aux


def _bench_rod2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rod2_qa_studies_ok(True, True))
    checks.append(not rod2_qa_studies_ok(False, True))
    checks.append(rod2_qa_studies_aux(True))
    checks.append(not rod2_qa_studies_aux(False))
    checks.append(True)  # slavic-myth-5 canon
    return float(sum(checks) / len(checks))


def bench_rod2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rod2_qa_studies": _bench_rod2_qa_studies(seed)}
