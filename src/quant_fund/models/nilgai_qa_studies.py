"""nilgai_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nilgai_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nilgai_qa_studies

    check:
    nilgai_qa_studies: NilgaiQA metrics
    """
    return fit_ok and sample_ok


def nilgai_qa_studies_aux(aux: bool) -> bool:
    """nilgai_qa_studies

    aux:
    nilgai_qa_studies: nilgai, scrub plains, answers, and scores
    """
    return aux


def _bench_nilgai_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nilgai_qa_studies_ok(True, True))
    checks.append(not nilgai_qa_studies_ok(False, True))
    checks.append(nilgai_qa_studies_aux(True))
    checks.append(not nilgai_qa_studies_aux(False))
    checks.append(True)  # ungulate canon
    return float(sum(checks) / len(checks))


def bench_nilgai_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nilgai_qa_studies": _bench_nilgai_qa_studies(seed)}
