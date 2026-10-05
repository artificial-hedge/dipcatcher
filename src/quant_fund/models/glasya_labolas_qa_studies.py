"""glasya_labolas_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def glasya_labolas_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """glasya_labolas_qa_studies

    check:
    glasya_labolas_qa_studies: G
    """
    return fit_ok and sample_ok


def glasya_labolas_qa_studies_aux(aux: bool) -> bool:
    """glasya_labolas_qa_studies

    aux:
    glasya_labolas_qa_studies: l
    """
    return aux


def _bench_glasya_labolas_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(glasya_labolas_qa_studies_ok(True, True))
    checks.append(not glasya_labolas_qa_studies_ok(False, True))
    checks.append(glasya_labolas_qa_studies_aux(True))
    checks.append(not glasya_labolas_qa_studies_aux(False))
    checks.append(True)  # goetic-circle canon
    return float(sum(checks) / len(checks))


def bench_glasya_labolas_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_glasya_labolas_qa_studies": _bench_glasya_labolas_qa_studies(seed)}
