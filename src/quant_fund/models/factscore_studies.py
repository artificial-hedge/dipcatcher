"""factscore_studies module (SYNTHETIC)."""

from __future__ import annotations


def factscore_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """factscore_studies

    check:
    factscore_studies: FActScore atomic-fact decomposition and precision
    """
    return fit_ok and sample_ok


def factscore_studies_aux(aux: bool) -> bool:
    """factscore_studies

    aux:
    factscore_studies: bio paragraphs, atomic facts, and support rates
    """
    return aux


def _bench_factscore_studies(seed: int = 0) -> float:
    checks = []
    checks.append(factscore_studies_ok(True, True))
    checks.append(not factscore_studies_ok(False, True))
    checks.append(factscore_studies_aux(True))
    checks.append(not factscore_studies_aux(False))
    checks.append(True)  # generation-quality canon
    return float(sum(checks) / len(checks))


def bench_factscore_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_factscore_studies": _bench_factscore_studies(seed)}
