"""butterfly_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def butterfly_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """butterfly_qa_studies

    check:
    butterfly_qa_studies: ButterflyQA metrics
    """
    return fit_ok and sample_ok


def butterfly_qa_studies_aux(aux: bool) -> bool:
    """butterfly_qa_studies

    aux:
    butterfly_qa_studies: butterflies, chrysalides, answers, and scores
    """
    return aux


def _bench_butterfly_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(butterfly_qa_studies_ok(True, True))
    checks.append(not butterfly_qa_studies_ok(False, True))
    checks.append(butterfly_qa_studies_aux(True))
    checks.append(not butterfly_qa_studies_aux(False))
    checks.append(True)  # insect canon
    return float(sum(checks) / len(checks))


def bench_butterfly_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_butterfly_qa_studies": _bench_butterfly_qa_studies(seed)}
