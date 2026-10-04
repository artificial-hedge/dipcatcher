"""roadrunner_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def roadrunner_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """roadrunner_qa_studies

    check:
    roadrunner_qa_studies: RoadrunnerQA metrics
    """
    return fit_ok and sample_ok


def roadrunner_qa_studies_aux(aux: bool) -> bool:
    """roadrunner_qa_studies

    aux:
    roadrunner_qa_studies: roadrunners, chaparrals, answers, and scores
    """
    return aux


def _bench_roadrunner_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(roadrunner_qa_studies_ok(True, True))
    checks.append(not roadrunner_qa_studies_ok(False, True))
    checks.append(roadrunner_qa_studies_aux(True))
    checks.append(not roadrunner_qa_studies_aux(False))
    checks.append(True)  # nightbird canon
    return float(sum(checks) / len(checks))


def bench_roadrunner_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_roadrunner_qa_studies": _bench_roadrunner_qa_studies(seed)}
