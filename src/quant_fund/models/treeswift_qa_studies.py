"""treeswift_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def treeswift_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """treeswift_qa_studies

    check:
    treeswift_qa_studies: TreeswiftQA metrics
    """
    return fit_ok and sample_ok


def treeswift_qa_studies_aux(aux: bool) -> bool:
    """treeswift_qa_studies

    aux:
    treeswift_qa_studies: treeswifts, perches, answers, and scores
    """
    return aux


def _bench_treeswift_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(treeswift_qa_studies_ok(True, True))
    checks.append(not treeswift_qa_studies_ok(False, True))
    checks.append(treeswift_qa_studies_aux(True))
    checks.append(not treeswift_qa_studies_aux(False))
    checks.append(True)  # aerialist canon
    return float(sum(checks) / len(checks))


def bench_treeswift_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_treeswift_qa_studies": _bench_treeswift_qa_studies(seed)}
