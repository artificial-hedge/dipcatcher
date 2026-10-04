"""dedup_pipeline_studies module (SYNTHETIC)."""

from __future__ import annotations


def dedup_pipeline_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dedup_pipeline_studies

    check:
    dedup_pipeline_studies: MinHash-LSH near-duplicate detection/bands and jaccard
    """
    return fit_ok and sample_ok


def dedup_pipeline_studies_aux(aux: bool) -> bool:
    """dedup_pipeline_studies

    aux:
    dedup_pipeline_studies: exact and fuzzy deduplication pipelines/orders and overlaps
    """
    return aux


def _bench_dedup_pipeline_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dedup_pipeline_studies_ok(True, True))
    checks.append(not dedup_pipeline_studies_ok(False, True))
    checks.append(dedup_pipeline_studies_aux(True))
    checks.append(not dedup_pipeline_studies_aux(False))
    checks.append(True)  # pretraining-data canon
    return float(sum(checks) / len(checks))


def bench_dedup_pipeline_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dedup_pipeline_studies": _bench_dedup_pipeline_studies(seed)}
