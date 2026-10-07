"""Wave-1363 bench adapters: rumor-bias canon (SYNTHETIC only)."""

from quant_fund.models import (
    age_bias_studies,
    curry_qa_studies,
    cw_qa2_studies,
    dialect_bias_studies,
    politi_fact_studies,
    rumor_twitter_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13630


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


def bench_age_bias_studies_family(seed: int = _SEED + 0):
    """age_bias_studies: synthetic correctness bench."""
    return _finite_blob(age_bias_studies.bench_age_bias_studies(seed))


def bench_curry_qa_studies_family(seed: int = _SEED + 1):
    """curry_qa_studies: synthetic correctness bench."""
    return _finite_blob(curry_qa_studies.bench_curry_qa_studies(seed))


def bench_cw_qa2_studies_family(seed: int = _SEED + 2):
    """cw_qa2_studies: synthetic correctness bench."""
    return _finite_blob(cw_qa2_studies.bench_cw_qa2_studies(seed))


def bench_dialect_bias_studies_family(seed: int = _SEED + 3):
    """dialect_bias_studies: synthetic correctness bench."""
    return _finite_blob(dialect_bias_studies.bench_dialect_bias_studies(seed))


def bench_politi_fact_studies_family(seed: int = _SEED + 4):
    """politi_fact_studies: synthetic correctness bench."""
    return _finite_blob(politi_fact_studies.bench_politi_fact_studies(seed))


def bench_rumor_twitter_studies_family(seed: int = _SEED + 5):
    """rumor_twitter_studies: synthetic correctness bench."""
    return _finite_blob(rumor_twitter_studies.bench_rumor_twitter_studies(seed))
