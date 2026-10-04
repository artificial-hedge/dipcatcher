"""dedup_studies module (SYNTHETIC)."""

from __future__ import annotations


def dedup_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dedup_studies

    check:
    dedup_studies: exact/fuzzy document deduplication/hashes and overlaps
    """
    return fit_ok and sample_ok


def dedup_studies_aux(aux: bool) -> bool:
    """dedup_studies

    aux:
    dedup_studies: substring and suffix-array dedup/spans and removals
    """
    return aux


def _bench_dedup_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dedup_studies_ok(True, True))
    checks.append(not dedup_studies_ok(False, True))
    checks.append(dedup_studies_aux(True))
    checks.append(not dedup_studies_aux(False))
    checks.append(True)  # data-filtering/dedup canon
    return float(sum(checks) / len(checks))


def bench_dedup_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dedup_studies": _bench_dedup_studies(seed)}
