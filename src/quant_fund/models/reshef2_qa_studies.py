"""reshef2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def reshef2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reshef2_qa_studies

    check:
    reshef2_qa_studies: Reshef2QA metrics
    """
    return fit_ok and sample_ok


def reshef2_qa_studies_aux(aux: bool) -> bool:
    """reshef2_qa_studies

    aux:
    reshef2_qa_studies: reshef2, plague archers, answers, and scores
    """
    return aux


def _bench_reshef2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(reshef2_qa_studies_ok(True, True))
    checks.append(not reshef2_qa_studies_ok(False, True))
    checks.append(reshef2_qa_studies_aux(True))
    checks.append(not reshef2_qa_studies_aux(False))
    checks.append(True)  # phoenician-2 canon
    return float(sum(checks) / len(checks))


def bench_reshef2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reshef2_qa_studies": _bench_reshef2_qa_studies(seed)}
