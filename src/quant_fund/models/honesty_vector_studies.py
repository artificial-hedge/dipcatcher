"""honesty_vector_studies module (SYNTHETIC)."""

from __future__ import annotations


def honesty_vector_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """honesty_vector_studies

    check:
    honesty_vector_studies: truthful-vs-deceptive direction mining/pairs and layers
    """
    return fit_ok and sample_ok


def honesty_vector_studies_aux(aux: bool) -> bool:
    """honesty_vector_studies

    aux:
    honesty_vector_studies: honesty-classifier steering signals/probes and contrasts
    """
    return aux


def _bench_honesty_vector_studies(seed: int = 0) -> float:
    checks = []
    checks.append(honesty_vector_studies_ok(True, True))
    checks.append(not honesty_vector_studies_ok(False, True))
    checks.append(honesty_vector_studies_aux(True))
    checks.append(not honesty_vector_studies_aux(False))
    checks.append(True)  # representation-engineering canon
    return float(sum(checks) / len(checks))


def bench_honesty_vector_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_honesty_vector_studies": _bench_honesty_vector_studies(seed)}
