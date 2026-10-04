"""nunbird_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nunbird_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nunbird_qa_studies

    check:
    nunbird_qa_studies: NunbirdQA metrics
    """
    return fit_ok and sample_ok


def nunbird_qa_studies_aux(aux: bool) -> bool:
    """nunbird_qa_studies

    aux:
    nunbird_qa_studies: nunbirds, clearings, answers, and scores
    """
    return aux


def _bench_nunbird_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nunbird_qa_studies_ok(True, True))
    checks.append(not nunbird_qa_studies_ok(False, True))
    checks.append(nunbird_qa_studies_aux(True))
    checks.append(not nunbird_qa_studies_aux(False))
    checks.append(True)  # coraciiform canon
    return float(sum(checks) / len(checks))


def bench_nunbird_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nunbird_qa_studies": _bench_nunbird_qa_studies(seed)}
