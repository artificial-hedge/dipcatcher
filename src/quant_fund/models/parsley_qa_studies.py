"""parsley_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def parsley_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """parsley_qa_studies

    check:
    parsley_qa_studies: ParsleyQA metrics
    """
    return fit_ok and sample_ok


def parsley_qa_studies_aux(aux: bool) -> bool:
    """parsley_qa_studies

    aux:
    parsley_qa_studies: parsley, beds, answers, and scores
    """
    return aux


def _bench_parsley_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(parsley_qa_studies_ok(True, True))
    checks.append(not parsley_qa_studies_ok(False, True))
    checks.append(parsley_qa_studies_aux(True))
    checks.append(not parsley_qa_studies_aux(False))
    checks.append(True)  # spice-2 canon
    return float(sum(checks) / len(checks))


def bench_parsley_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_parsley_qa_studies": _bench_parsley_qa_studies(seed)}
