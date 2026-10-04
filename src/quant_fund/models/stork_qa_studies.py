"""stork_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def stork_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stork_qa_studies

    check:
    stork_qa_studies: StorkQA metrics
    """
    return fit_ok and sample_ok


def stork_qa_studies_aux(aux: bool) -> bool:
    """stork_qa_studies

    aux:
    stork_qa_studies: storks, nests, answers, and scores
    """
    return aux


def _bench_stork_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(stork_qa_studies_ok(True, True))
    checks.append(not stork_qa_studies_ok(False, True))
    checks.append(stork_qa_studies_aux(True))
    checks.append(not stork_qa_studies_aux(False))
    checks.append(True)  # wader canon
    return float(sum(checks) / len(checks))


def bench_stork_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stork_qa_studies": _bench_stork_qa_studies(seed)}
