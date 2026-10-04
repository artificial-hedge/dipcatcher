"""concept_vector_studies module (SYNTHETIC)."""

from __future__ import annotations


def concept_vector_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """concept_vector_studies

    check:
    concept_vector_studies: linear concept extraction/directions and classes
    """
    return fit_ok and sample_ok


def concept_vector_studies_aux(aux: bool) -> bool:
    """concept_vector_studies

    aux:
    concept_vector_studies: latent concept arithmetic and search/vectors and tokens
    """
    return aux


def _bench_concept_vector_studies(seed: int = 0) -> float:
    checks = []
    checks.append(concept_vector_studies_ok(True, True))
    checks.append(not concept_vector_studies_ok(False, True))
    checks.append(concept_vector_studies_aux(True))
    checks.append(not concept_vector_studies_aux(False))
    checks.append(True)  # representation-engineering canon
    return float(sum(checks) / len(checks))


def bench_concept_vector_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_concept_vector_studies": _bench_concept_vector_studies(seed)}
