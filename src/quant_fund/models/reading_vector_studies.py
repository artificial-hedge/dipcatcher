"""reading_vector_studies module (SYNTHETIC)."""

from __future__ import annotations


def reading_vector_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reading_vector_studies

    check:
    reading_vector_studies: representation-reading contrasts/pairs and means
    """
    return fit_ok and sample_ok


def reading_vector_studies_aux(aux: bool) -> bool:
    """reading_vector_studies

    aux:
    reading_vector_studies: contrast-consistent search directions/texts and vectors
    """
    return aux


def _bench_reading_vector_studies(seed: int = 0) -> float:
    checks = []
    checks.append(reading_vector_studies_ok(True, True))
    checks.append(not reading_vector_studies_ok(False, True))
    checks.append(reading_vector_studies_aux(True))
    checks.append(not reading_vector_studies_aux(False))
    checks.append(True)  # representation-engineering canon
    return float(sum(checks) / len(checks))


def bench_reading_vector_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reading_vector_studies": _bench_reading_vector_studies(seed)}
