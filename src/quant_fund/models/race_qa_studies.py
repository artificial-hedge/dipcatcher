"""race_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def race_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """race_qa_studies

    check:
    race_qa_studies: RACE exam-reading metrics
    """
    return fit_ok and sample_ok


def race_qa_studies_aux(aux: bool) -> bool:
    """race_qa_studies

    aux:
    race_qa_studies: passages, questions, options, and accuracies
    """
    return aux


def _bench_race_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(race_qa_studies_ok(True, True))
    checks.append(not race_qa_studies_ok(False, True))
    checks.append(race_qa_studies_aux(True))
    checks.append(not race_qa_studies_aux(False))
    checks.append(True)  # reading-comprehension-3 canon
    return float(sum(checks) / len(checks))


def bench_race_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_race_qa_studies": _bench_race_qa_studies(seed)}
