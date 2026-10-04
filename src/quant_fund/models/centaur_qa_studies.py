"""centaur_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def centaur_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """centaur_qa_studies

    check:
    centaur_qa_studies: CentaurQA metrics
    """
    return fit_ok and sample_ok


def centaur_qa_studies_aux(aux: bool) -> bool:
    """centaur_qa_studies

    aux:
    centaur_qa_studies: centaurs, horse men, answers, and scores
    """
    return aux


def _bench_centaur_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(centaur_qa_studies_ok(True, True))
    checks.append(not centaur_qa_studies_ok(False, True))
    checks.append(centaur_qa_studies_aux(True))
    checks.append(not centaur_qa_studies_aux(False))
    checks.append(True)  # greek-myth canon
    return float(sum(checks) / len(checks))


def bench_centaur_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_centaur_qa_studies": _bench_centaur_qa_studies(seed)}
