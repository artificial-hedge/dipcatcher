"""clubmoss_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def clubmoss_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """clubmoss_qa_studies

    check:
    clubmoss_qa_studies: ClubmossQA metrics
    """
    return fit_ok and sample_ok


def clubmoss_qa_studies_aux(aux: bool) -> bool:
    """clubmoss_qa_studies

    aux:
    clubmoss_qa_studies: clubmosses, forests, answers, and scores
    """
    return aux


def _bench_clubmoss_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(clubmoss_qa_studies_ok(True, True))
    checks.append(not clubmoss_qa_studies_ok(False, True))
    checks.append(clubmoss_qa_studies_aux(True))
    checks.append(not clubmoss_qa_studies_aux(False))
    checks.append(True)  # moss canon
    return float(sum(checks) / len(checks))


def bench_clubmoss_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_clubmoss_qa_studies": _bench_clubmoss_qa_studies(seed)}
