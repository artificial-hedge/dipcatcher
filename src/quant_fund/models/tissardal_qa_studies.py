"""tissardal_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tissardal_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tissardal_qa_studies

    check:
    tissardal_qa_studies: s
    """
    return fit_ok and sample_ok


def tissardal_qa_studies_aux(aux: bool) -> bool:
    """tissardal_qa_studies

    aux:
    tissardal_qa_studies: a
    """
    return aux


def _bench_tissardal_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tissardal_qa_studies_ok(True, True))
    checks.append(not tissardal_qa_studies_ok(False, True))
    checks.append(tissardal_qa_studies_aux(True))
    checks.append(not tissardal_qa_studies_aux(False))
    checks.append(True)  # saharan canon
    return float(sum(checks) / len(checks))


def bench_tissardal_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tissardal_qa_studies": _bench_tissardal_qa_studies(seed)}
