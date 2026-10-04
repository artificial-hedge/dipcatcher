"""needle_multi_studies module (SYNTHETIC)."""

from __future__ import annotations


def needle_multi_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """needle_multi_studies

    check:
    needle_multi_studies: Multi-needle retrieval metrics
    """
    return fit_ok and sample_ok


def needle_multi_studies_aux(aux: bool) -> bool:
    """needle_multi_studies

    aux:
    needle_multi_studies: documents, keys, retrievals, and hit rates
    """
    return aux


def _bench_needle_multi_studies(seed: int = 0) -> float:
    checks = []
    checks.append(needle_multi_studies_ok(True, True))
    checks.append(not needle_multi_studies_ok(False, True))
    checks.append(needle_multi_studies_aux(True))
    checks.append(not needle_multi_studies_aux(False))
    checks.append(True)  # long-context-3 canon
    return float(sum(checks) / len(checks))


def bench_needle_multi_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_needle_multi_studies": _bench_needle_multi_studies(seed)}
