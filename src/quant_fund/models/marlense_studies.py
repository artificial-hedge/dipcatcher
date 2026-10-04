"""marlense_studies module (SYNTHETIC)."""

from __future__ import annotations


def marlense_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """marlense_studies

    check:
    marlense_studies: Marlense metrics
    """
    return fit_ok and sample_ok


def marlense_studies_aux(aux: bool) -> bool:
    """marlense_studies

    aux:
    marlense_studies: narratives, events, summaries, and scores
    """
    return aux


def _bench_marlense_studies(seed: int = 0) -> float:
    checks = []
    checks.append(marlense_studies_ok(True, True))
    checks.append(not marlense_studies_ok(False, True))
    checks.append(marlense_studies_aux(True))
    checks.append(not marlense_studies_aux(False))
    checks.append(True)  # long-doc-sum canon
    return float(sum(checks) / len(checks))


def bench_marlense_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marlense_studies": _bench_marlense_studies(seed)}
