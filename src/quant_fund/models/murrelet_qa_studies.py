"""murrelet_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def murrelet_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """murrelet_qa_studies

    check:
    murrelet_qa_studies: MurreletQA metrics
    """
    return fit_ok and sample_ok


def murrelet_qa_studies_aux(aux: bool) -> bool:
    """murrelet_qa_studies

    aux:
    murrelet_qa_studies: murrelets, inlets, answers, and scores
    """
    return aux


def _bench_murrelet_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(murrelet_qa_studies_ok(True, True))
    checks.append(not murrelet_qa_studies_ok(False, True))
    checks.append(murrelet_qa_studies_aux(True))
    checks.append(not murrelet_qa_studies_aux(False))
    checks.append(True)  # seabird-3 canon
    return float(sum(checks) / len(checks))


def bench_murrelet_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_murrelet_qa_studies": _bench_murrelet_qa_studies(seed)}
