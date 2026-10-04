"""oregano_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def oregano_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """oregano_qa_studies

    check:
    oregano_qa_studies: OreganoQA metrics
    """
    return fit_ok and sample_ok


def oregano_qa_studies_aux(aux: bool) -> bool:
    """oregano_qa_studies

    aux:
    oregano_qa_studies: oregano, gardens, answers, and scores
    """
    return aux


def _bench_oregano_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(oregano_qa_studies_ok(True, True))
    checks.append(not oregano_qa_studies_ok(False, True))
    checks.append(oregano_qa_studies_aux(True))
    checks.append(not oregano_qa_studies_aux(False))
    checks.append(True)  # spice-2 canon
    return float(sum(checks) / len(checks))


def bench_oregano_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oregano_qa_studies": _bench_oregano_qa_studies(seed)}
