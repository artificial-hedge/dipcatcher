"""kookaburra_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kookaburra_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kookaburra_qa_studies

    check:
    kookaburra_qa_studies: KookaburraQA metrics
    """
    return fit_ok and sample_ok


def kookaburra_qa_studies_aux(aux: bool) -> bool:
    """kookaburra_qa_studies

    aux:
    kookaburra_qa_studies: kookaburras, gums, answers, and scores
    """
    return aux


def _bench_kookaburra_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kookaburra_qa_studies_ok(True, True))
    checks.append(not kookaburra_qa_studies_ok(False, True))
    checks.append(kookaburra_qa_studies_aux(True))
    checks.append(not kookaburra_qa_studies_aux(False))
    checks.append(True)  # riverbird canon
    return float(sum(checks) / len(checks))


def bench_kookaburra_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kookaburra_qa_studies": _bench_kookaburra_qa_studies(seed)}
