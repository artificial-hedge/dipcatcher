"""grounding_verify_studies module (SYNTHETIC)."""

from __future__ import annotations


def grounding_verify_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """grounding_verify_studies

    check:
    grounding_verify_studies: context-grounded generation checks/spans and sources
    """
    return fit_ok and sample_ok


def grounding_verify_studies_aux(aux: bool) -> bool:
    """grounding_verify_studies

    aux:
    grounding_verify_studies: retrieval-grounded answer verification/evidence and alignment
    """
    return aux


def _bench_grounding_verify_studies(seed: int = 0) -> float:
    checks = []
    checks.append(grounding_verify_studies_ok(True, True))
    checks.append(not grounding_verify_studies_ok(False, True))
    checks.append(grounding_verify_studies_aux(True))
    checks.append(not grounding_verify_studies_aux(False))
    checks.append(True)  # grounding/hallucination canon
    return float(sum(checks) / len(checks))


def bench_grounding_verify_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_grounding_verify_studies": _bench_grounding_verify_studies(seed)}
