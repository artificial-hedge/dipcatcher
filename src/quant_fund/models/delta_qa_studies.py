"""delta_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def delta_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """delta_qa_studies

    check:
    delta_qa_studies: DeltaQA metrics
    """
    return fit_ok and sample_ok


def delta_qa_studies_aux(aux: bool) -> bool:
    """delta_qa_studies

    aux:
    delta_qa_studies: deltas, channels, answers, and scores
    """
    return aux


def _bench_delta_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(delta_qa_studies_ok(True, True))
    checks.append(not delta_qa_studies_ok(False, True))
    checks.append(delta_qa_studies_aux(True))
    checks.append(not delta_qa_studies_aux(False))
    checks.append(True)  # wetland canon
    return float(sum(checks) / len(checks))


def bench_delta_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_delta_qa_studies": _bench_delta_qa_studies(seed)}
