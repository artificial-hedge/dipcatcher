"""mango_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mango_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mango_qa_studies

    check:
    mango_qa_studies: MangoQA metrics
    """
    return fit_ok and sample_ok


def mango_qa_studies_aux(aux: bool) -> bool:
    """mango_qa_studies

    aux:
    mango_qa_studies: mangos, pulp, answers, and scores
    """
    return aux


def _bench_mango_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mango_qa_studies_ok(True, True))
    checks.append(not mango_qa_studies_ok(False, True))
    checks.append(mango_qa_studies_aux(True))
    checks.append(not mango_qa_studies_aux(False))
    checks.append(True)  # fruit canon
    return float(sum(checks) / len(checks))


def bench_mango_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mango_qa_studies": _bench_mango_qa_studies(seed)}
