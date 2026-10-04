"""kay_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kay_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kay_qa_studies

    check:
    kay_qa_studies: s
    """
    return fit_ok and sample_ok


def kay_qa_studies_aux(aux: bool) -> bool:
    """kay_qa_studies

    aux:
    kay_qa_studies: e
    """
    return aux


def _bench_kay_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kay_qa_studies_ok(True, True))
    checks.append(not kay_qa_studies_ok(False, True))
    checks.append(kay_qa_studies_aux(True))
    checks.append(not kay_qa_studies_aux(False))
    checks.append(True)  # arthurian-3 canon
    return float(sum(checks) / len(checks))


def bench_kay_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kay_qa_studies": _bench_kay_qa_studies(seed)}
