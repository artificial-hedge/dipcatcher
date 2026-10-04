"""Wave-1294 bench adapters: representation-engineering canon (SYNTHETIC only)."""

from quant_fund.models import (
    activation_oracle_studies,
    concept_vector_studies,
    feature_ablation_studies,
    honesty_vector_studies,
    reading_vector_studies,
    refusal_vector_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12940


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_activation_oracle_studies_family(seed: int = _SEED + 0):
    """activation_oracle_studies: synthetic correctness bench."""
    return _finite_blob(activation_oracle_studies.bench_activation_oracle_studies(seed))


def bench_concept_vector_studies_family(seed: int = _SEED + 1):
    """concept_vector_studies: synthetic correctness bench."""
    return _finite_blob(concept_vector_studies.bench_concept_vector_studies(seed))


def bench_feature_ablation_studies_family(seed: int = _SEED + 2):
    """feature_ablation_studies: synthetic correctness bench."""
    return _finite_blob(feature_ablation_studies.bench_feature_ablation_studies(seed))


def bench_honesty_vector_studies_family(seed: int = _SEED + 3):
    """honesty_vector_studies: synthetic correctness bench."""
    return _finite_blob(honesty_vector_studies.bench_honesty_vector_studies(seed))


def bench_reading_vector_studies_family(seed: int = _SEED + 4):
    """reading_vector_studies: synthetic correctness bench."""
    return _finite_blob(reading_vector_studies.bench_reading_vector_studies(seed))


def bench_refusal_vector_studies_family(seed: int = _SEED + 5):
    """refusal_vector_studies: synthetic correctness bench."""
    return _finite_blob(refusal_vector_studies.bench_refusal_vector_studies(seed))
