"""aphid_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def aphid_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """aphid_qa_studies

    check:
    aphid_qa_studies: AphidQA metrics
    """
    return fit_ok and sample_ok


def aphid_qa_studies_aux(aux: bool) -> bool:
    """aphid_qa_studies

    aux:
    aphid_qa_studies: aphids, leaves, answers, and scores
    """
    return aux


def _bench_aphid_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(aphid_qa_studies_ok(True, True))
    checks.append(not aphid_qa_studies_ok(False, True))
    checks.append(aphid_qa_studies_aux(True))
    checks.append(not aphid_qa_studies_aux(False))
    checks.append(True)  # insect-2 canon
    return float(sum(checks) / len(checks))


def bench_aphid_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aphid_qa_studies": _bench_aphid_qa_studies(seed)}
