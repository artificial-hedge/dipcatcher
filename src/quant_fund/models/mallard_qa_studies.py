"""mallard_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mallard_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mallard_qa_studies

    check:
    mallard_qa_studies: MallardQA metrics
    """
    return fit_ok and sample_ok


def mallard_qa_studies_aux(aux: bool) -> bool:
    """mallard_qa_studies

    aux:
    mallard_qa_studies: mallards, wetlands, answers, and scores
    """
    return aux


def _bench_mallard_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mallard_qa_studies_ok(True, True))
    checks.append(not mallard_qa_studies_ok(False, True))
    checks.append(mallard_qa_studies_aux(True))
    checks.append(not mallard_qa_studies_aux(False))
    checks.append(True)  # duck canon
    return float(sum(checks) / len(checks))


def bench_mallard_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mallard_qa_studies": _bench_mallard_qa_studies(seed)}
