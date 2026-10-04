"""nunlet_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nunlet_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nunlet_qa_studies

    check:
    nunlet_qa_studies: NunletQA metrics
    """
    return fit_ok and sample_ok


def nunlet_qa_studies_aux(aux: bool) -> bool:
    """nunlet_qa_studies

    aux:
    nunlet_qa_studies: nunlets, vine tangles, answers, and scores
    """
    return aux


def _bench_nunlet_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nunlet_qa_studies_ok(True, True))
    checks.append(not nunlet_qa_studies_ok(False, True))
    checks.append(nunlet_qa_studies_aux(True))
    checks.append(not nunlet_qa_studies_aux(False))
    checks.append(True)  # coraciiform canon
    return float(sum(checks) / len(checks))


def bench_nunlet_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nunlet_qa_studies": _bench_nunlet_qa_studies(seed)}
