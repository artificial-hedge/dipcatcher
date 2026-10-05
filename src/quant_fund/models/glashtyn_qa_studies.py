"""glashtyn_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def glashtyn_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """glashtyn_qa_studies

    check:
    glashtyn_qa_studies: w
    """
    return fit_ok and sample_ok


def glashtyn_qa_studies_aux(aux: bool) -> bool:
    """glashtyn_qa_studies

    aux:
    glashtyn_qa_studies: a
    """
    return aux


def _bench_glashtyn_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(glashtyn_qa_studies_ok(True, True))
    checks.append(not glashtyn_qa_studies_ok(False, True))
    checks.append(glashtyn_qa_studies_aux(True))
    checks.append(not glashtyn_qa_studies_aux(False))
    checks.append(True)  # manx-myth canon
    return float(sum(checks) / len(checks))


def bench_glashtyn_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_glashtyn_qa_studies": _bench_glashtyn_qa_studies(seed)}
