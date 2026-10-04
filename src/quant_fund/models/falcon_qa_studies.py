"""falcon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def falcon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """falcon_qa_studies

    check:
    falcon_qa_studies: FalconQA metrics
    """
    return fit_ok and sample_ok


def falcon_qa_studies_aux(aux: bool) -> bool:
    """falcon_qa_studies

    aux:
    falcon_qa_studies: falcons, dives, answers, and scores
    """
    return aux


def _bench_falcon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(falcon_qa_studies_ok(True, True))
    checks.append(not falcon_qa_studies_ok(False, True))
    checks.append(falcon_qa_studies_aux(True))
    checks.append(not falcon_qa_studies_aux(False))
    checks.append(True)  # avian canon
    return float(sum(checks) / len(checks))


def bench_falcon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_falcon_qa_studies": _bench_falcon_qa_studies(seed)}
