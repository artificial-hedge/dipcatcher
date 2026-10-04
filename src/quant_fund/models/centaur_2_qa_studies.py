"""centaur_2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def centaur_2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """centaur_2_qa_studies

    check:
    centaur_2_qa_studies: Centaur2QA metrics
    """
    return fit_ok and sample_ok


def centaur_2_qa_studies_aux(aux: bool) -> bool:
    """centaur_2_qa_studies

    aux:
    centaur_2_qa_studies: centaurs, archer plains, answers, and scores
    """
    return aux


def _bench_centaur_2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(centaur_2_qa_studies_ok(True, True))
    checks.append(not centaur_2_qa_studies_ok(False, True))
    checks.append(centaur_2_qa_studies_aux(True))
    checks.append(not centaur_2_qa_studies_aux(False))
    checks.append(True)  # greek-beast canon
    return float(sum(checks) / len(checks))


def bench_centaur_2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_centaur_2_qa_studies": _bench_centaur_2_qa_studies(seed)}
