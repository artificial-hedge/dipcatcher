"""topi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def topi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """topi_qa_studies

    check:
    topi_qa_studies: TopiQA metrics
    """
    return fit_ok and sample_ok


def topi_qa_studies_aux(aux: bool) -> bool:
    """topi_qa_studies

    aux:
    topi_qa_studies: topis, grasslands, answers, and scores
    """
    return aux


def _bench_topi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(topi_qa_studies_ok(True, True))
    checks.append(not topi_qa_studies_ok(False, True))
    checks.append(topi_qa_studies_aux(True))
    checks.append(not topi_qa_studies_aux(False))
    checks.append(True)  # antelope-2 canon
    return float(sum(checks) / len(checks))


def bench_topi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_topi_qa_studies": _bench_topi_qa_studies(seed)}
