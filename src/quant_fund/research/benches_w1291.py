"""Wave-1291 bench adapters: data-filtering/dedup canon (SYNTHETIC only)."""

from quant_fund.models import (
    data_mix_studies,
    dedup_minhash_studies,
    dedup_studies,
    domain_classifier_studies,
    perplexity_filter_studies,
    quality_filter_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12910


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_data_mix_studies_family(seed: int = _SEED + 0):
    """data_mix_studies: synthetic correctness bench."""
    return _finite_blob(data_mix_studies.bench_data_mix_studies(seed))


def bench_dedup_minhash_studies_family(seed: int = _SEED + 1):
    """dedup_minhash_studies: synthetic correctness bench."""
    return _finite_blob(dedup_minhash_studies.bench_dedup_minhash_studies(seed))


def bench_dedup_studies_family(seed: int = _SEED + 2):
    """dedup_studies: synthetic correctness bench."""
    return _finite_blob(dedup_studies.bench_dedup_studies(seed))


def bench_domain_classifier_studies_family(seed: int = _SEED + 3):
    """domain_classifier_studies: synthetic correctness bench."""
    return _finite_blob(domain_classifier_studies.bench_domain_classifier_studies(seed))


def bench_perplexity_filter_studies_family(seed: int = _SEED + 4):
    """perplexity_filter_studies: synthetic correctness bench."""
    return _finite_blob(perplexity_filter_studies.bench_perplexity_filter_studies(seed))


def bench_quality_filter_studies_family(seed: int = _SEED + 5):
    """quality_filter_studies: synthetic correctness bench."""
    return _finite_blob(quality_filter_studies.bench_quality_filter_studies(seed))
