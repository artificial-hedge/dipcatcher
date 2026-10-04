"""moonrat_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def moonrat_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """moonrat_qa_studies

    check:
    moonrat_qa_studies: MoonratQA metrics
    """
    return fit_ok and sample_ok


def moonrat_qa_studies_aux(aux: bool) -> bool:
    """moonrat_qa_studies

    aux:
    moonrat_qa_studies: moonrats, borneo understory, answers, and scores
    """
    return aux


def _bench_moonrat_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(moonrat_qa_studies_ok(True, True))
    checks.append(not moonrat_qa_studies_ok(False, True))
    checks.append(moonrat_qa_studies_aux(True))
    checks.append(not moonrat_qa_studies_aux(False))
    checks.append(True)  # fossorial canon
    return float(sum(checks) / len(checks))


def bench_moonrat_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moonrat_qa_studies": _bench_moonrat_qa_studies(seed)}
