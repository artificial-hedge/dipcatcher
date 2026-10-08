"""Wave-1265 bench adapters: genetic-epidemiology/MR canon (SYNTHETIC only)."""

from quant_fund.models import (
    colocalization_studies,
    genetic_correlation_studies,
    heritability_ldscore_studies,
    mendelian_randomization_studies,
    pleiotropy_robust_studies,
    polygenic_score_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12650


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_colocalization_studies_family(seed: int = _SEED + 0):
    """colocalization_studies: synthetic correctness bench."""
    return _finite_blob(colocalization_studies.bench_colocalization_studies(seed))


def bench_genetic_correlation_studies_family(seed: int = _SEED + 1):
    """genetic_correlation_studies: synthetic correctness bench."""
    return _finite_blob(genetic_correlation_studies.bench_genetic_correlation_studies(seed))


def bench_heritability_ldscore_studies_family(seed: int = _SEED + 2):
    """heritability_ldscore_studies: synthetic correctness bench."""
    return _finite_blob(heritability_ldscore_studies.bench_heritability_ldscore_studies(seed))


def bench_mendelian_randomization_studies_family(seed: int = _SEED + 3):
    """mendelian_randomization_studies: synthetic correctness bench."""
    return _finite_blob(mendelian_randomization_studies.bench_mendelian_randomization_studies(seed))


def bench_pleiotropy_robust_studies_family(seed: int = _SEED + 4):
    """pleiotropy_robust_studies: synthetic correctness bench."""
    return _finite_blob(pleiotropy_robust_studies.bench_pleiotropy_robust_studies(seed))


def bench_polygenic_score_studies_family(seed: int = _SEED + 5):
    """polygenic_score_studies: synthetic correctness bench."""
    return _finite_blob(polygenic_score_studies.bench_polygenic_score_studies(seed))
