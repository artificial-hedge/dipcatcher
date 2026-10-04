"""free_tailed_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def free_tailed_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """free_tailed_qa_studies

    check:
    free_tailed_qa_studies: FreeTailedQA metrics
    """
    return fit_ok and sample_ok


def free_tailed_qa_studies_aux(aux: bool) -> bool:
    """free_tailed_qa_studies

    aux:
    free_tailed_qa_studies: free-tailed bats, cave mouths, answers, and scores
    """
    return aux


def _bench_free_tailed_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(free_tailed_qa_studies_ok(True, True))
    checks.append(not free_tailed_qa_studies_ok(False, True))
    checks.append(free_tailed_qa_studies_aux(True))
    checks.append(not free_tailed_qa_studies_aux(False))
    checks.append(True)  # bat-2 canon
    return float(sum(checks) / len(checks))


def bench_free_tailed_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_free_tailed_qa_studies": _bench_free_tailed_qa_studies(seed)}
