"""drill_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def drill_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """drill_qa_studies

    check:
    drill_qa_studies: DrillQA metrics
    """
    return fit_ok and sample_ok


def drill_qa_studies_aux(aux: bool) -> bool:
    """drill_qa_studies

    aux:
    drill_qa_studies: drills, coastal forests, answers, and scores
    """
    return aux


def _bench_drill_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(drill_qa_studies_ok(True, True))
    checks.append(not drill_qa_studies_ok(False, True))
    checks.append(drill_qa_studies_aux(True))
    checks.append(not drill_qa_studies_aux(False))
    checks.append(True)  # old-world-monkey canon
    return float(sum(checks) / len(checks))


def bench_drill_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_drill_qa_studies": _bench_drill_qa_studies(seed)}
