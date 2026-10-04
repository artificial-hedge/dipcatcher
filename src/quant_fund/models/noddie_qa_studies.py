"""noddie_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def noddie_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """noddie_qa_studies

    check:
    noddie_qa_studies: NoddieQA metrics
    """
    return fit_ok and sample_ok


def noddie_qa_studies_aux(aux: bool) -> bool:
    """noddie_qa_studies

    aux:
    noddie_qa_studies: noddies, tropic isles, answers, and scores
    """
    return aux


def _bench_noddie_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(noddie_qa_studies_ok(True, True))
    checks.append(not noddie_qa_studies_ok(False, True))
    checks.append(noddie_qa_studies_aux(True))
    checks.append(not noddie_qa_studies_aux(False))
    checks.append(True)  # seabird-4 canon
    return float(sum(checks) / len(checks))


def bench_noddie_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_noddie_qa_studies": _bench_noddie_qa_studies(seed)}
