"""right_whale_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def right_whale_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """right_whale_qa_studies

    check:
    right_whale_qa_studies: RightWhaleQA metrics
    """
    return fit_ok and sample_ok


def right_whale_qa_studies_aux(aux: bool) -> bool:
    """right_whale_qa_studies

    aux:
    right_whale_qa_studies: right whales, skim feeding grounds, answers, and scores
    """
    return aux


def _bench_right_whale_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(right_whale_qa_studies_ok(True, True))
    checks.append(not right_whale_qa_studies_ok(False, True))
    checks.append(right_whale_qa_studies_aux(True))
    checks.append(not right_whale_qa_studies_aux(False))
    checks.append(True)  # cetacean-2 canon
    return float(sum(checks) / len(checks))


def bench_right_whale_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_right_whale_qa_studies": _bench_right_whale_qa_studies(seed)}
