"""demeter2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def demeter2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """demeter2_qa_studies

    check:
    demeter2_qa_studies: Demeter2QA metrics
    """
    return fit_ok and sample_ok


def demeter2_qa_studies_aux(aux: bool) -> bool:
    """demeter2_qa_studies

    aux:
    demeter2_qa_studies: demeter2, grain mothers, answers, and scores
    """
    return aux


def _bench_demeter2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(demeter2_qa_studies_ok(True, True))
    checks.append(not demeter2_qa_studies_ok(False, True))
    checks.append(demeter2_qa_studies_aux(True))
    checks.append(not demeter2_qa_studies_aux(False))
    checks.append(True)  # greek-myth-11 canon
    return float(sum(checks) / len(checks))


def bench_demeter2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_demeter2_qa_studies": _bench_demeter2_qa_studies(seed)}
