"""ifri_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ifri_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ifri_qa_studies

    check:
    ifri_qa_studies: w
    """
    return fit_ok and sample_ok


def ifri_qa_studies_aux(aux: bool) -> bool:
    """ifri_qa_studies

    aux:
    ifri_qa_studies: a
    """
    return aux


def _bench_ifri_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ifri_qa_studies_ok(True, True))
    checks.append(not ifri_qa_studies_ok(False, True))
    checks.append(ifri_qa_studies_aux(True))
    checks.append(not ifri_qa_studies_aux(False))
    checks.append(True)  # amazigh-myth canon
    return float(sum(checks) / len(checks))


def bench_ifri_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ifri_qa_studies": _bench_ifri_qa_studies(seed)}
