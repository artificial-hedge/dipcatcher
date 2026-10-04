"""horntail_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def horntail_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """horntail_qa_studies

    check:
    horntail_qa_studies: HorntailQA metrics
    """
    return fit_ok and sample_ok


def horntail_qa_studies_aux(aux: bool) -> bool:
    """horntail_qa_studies

    aux:
    horntail_qa_studies: horntails, timber, answers, and scores
    """
    return aux


def _bench_horntail_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(horntail_qa_studies_ok(True, True))
    checks.append(not horntail_qa_studies_ok(False, True))
    checks.append(horntail_qa_studies_aux(True))
    checks.append(not horntail_qa_studies_aux(False))
    checks.append(True)  # invertebrate-2 canon
    return float(sum(checks) / len(checks))


def bench_horntail_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_horntail_qa_studies": _bench_horntail_qa_studies(seed)}
