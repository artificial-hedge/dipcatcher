"""ngram_overlap_studies module (SYNTHETIC)."""

from __future__ import annotations


def ngram_overlap_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ngram_overlap_studies

    check:
    ngram_overlap_studies: N-gram overlap leakage metrics
    """
    return fit_ok and sample_ok


def ngram_overlap_studies_aux(aux: bool) -> bool:
    """ngram_overlap_studies

    aux:
    ngram_overlap_studies: samples, references, overlaps, and leak rates
    """
    return aux


def _bench_ngram_overlap_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ngram_overlap_studies_ok(True, True))
    checks.append(not ngram_overlap_studies_ok(False, True))
    checks.append(ngram_overlap_studies_aux(True))
    checks.append(not ngram_overlap_studies_aux(False))
    checks.append(True)  # eval-tooling canon
    return float(sum(checks) / len(checks))


def bench_ngram_overlap_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ngram_overlap_studies": _bench_ngram_overlap_studies(seed)}
