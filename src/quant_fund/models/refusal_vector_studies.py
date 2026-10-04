"""refusal_vector_studies module (SYNTHETIC)."""

from __future__ import annotations


def refusal_vector_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """refusal_vector_studies

    check:
    refusal_vector_studies: refusal-direction extraction and scaling/harmful and harmless
    """
    return fit_ok and sample_ok


def refusal_vector_studies_aux(aux: bool) -> bool:
    """refusal_vector_studies

    aux:
    refusal_vector_studies: refusal-circuit ablation studies/heads and shifts
    """
    return aux


def _bench_refusal_vector_studies(seed: int = 0) -> float:
    checks = []
    checks.append(refusal_vector_studies_ok(True, True))
    checks.append(not refusal_vector_studies_ok(False, True))
    checks.append(refusal_vector_studies_aux(True))
    checks.append(not refusal_vector_studies_aux(False))
    checks.append(True)  # representation-engineering canon
    return float(sum(checks) / len(checks))


def bench_refusal_vector_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_refusal_vector_studies": _bench_refusal_vector_studies(seed)}
