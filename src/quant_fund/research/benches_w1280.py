"""Wave-1280 bench adapters: pretraining-data canon (SYNTHETIC only)."""

from quant_fund.models import (
    data_mixture_studies,
    data_quality_studies,
    dedup_pipeline_studies,
    domain_filtering_studies,
    synthetic_data_studies,
    token_budget_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12800


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


def bench_data_mixture_studies_family(seed: int = _SEED + 0):
    """data_mixture_studies: synthetic correctness bench."""
    return _finite_blob(data_mixture_studies.bench_data_mixture_studies(seed))


def bench_data_quality_studies_family(seed: int = _SEED + 1):
    """data_quality_studies: synthetic correctness bench."""
    return _finite_blob(data_quality_studies.bench_data_quality_studies(seed))


def bench_dedup_pipeline_studies_family(seed: int = _SEED + 2):
    """dedup_pipeline_studies: synthetic correctness bench."""
    return _finite_blob(dedup_pipeline_studies.bench_dedup_pipeline_studies(seed))


def bench_domain_filtering_studies_family(seed: int = _SEED + 3):
    """domain_filtering_studies: synthetic correctness bench."""
    return _finite_blob(domain_filtering_studies.bench_domain_filtering_studies(seed))


def bench_synthetic_data_studies_family(seed: int = _SEED + 4):
    """synthetic_data_studies: synthetic correctness bench."""
    return _finite_blob(synthetic_data_studies.bench_synthetic_data_studies(seed))


def bench_token_budget_studies_family(seed: int = _SEED + 5):
    """token_budget_studies: synthetic correctness bench."""
    return _finite_blob(token_budget_studies.bench_token_budget_studies(seed))
