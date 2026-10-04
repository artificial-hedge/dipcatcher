"""canyon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def canyon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """canyon_qa_studies

    check:
    canyon_qa_studies: CanyonQA metrics
    """
    return fit_ok and sample_ok


def canyon_qa_studies_aux(aux: bool) -> bool:
    """canyon_qa_studies

    aux:
    canyon_qa_studies: canyons, features, answers, and scores
    """
    return aux


def _bench_canyon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(canyon_qa_studies_ok(True, True))
    checks.append(not canyon_qa_studies_ok(False, True))
    checks.append(canyon_qa_studies_aux(True))
    checks.append(not canyon_qa_studies_aux(False))
    checks.append(True)  # terrain-2 canon
    return float(sum(checks) / len(checks))


def bench_canyon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_canyon_qa_studies": _bench_canyon_qa_studies(seed)}
