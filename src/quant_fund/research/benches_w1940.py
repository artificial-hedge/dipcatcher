"""Wave-1940 bench adapters: burmese-nat canon (SYNTHETIC only)."""

from quant_fund.models import (
    magami_qa_studies,
    mahagiri_qa_studies,
    min_kyawzwa_qa_studies,
    shwe_nabay_qa_studies,
    taungmagyi_qa_studies,
    thagya_min_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19400


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_magami_qa_studies_family(seed: int = _SEED + 0):
    """magami_qa_studies: synthetic correctness bench."""
    return _finite_blob(magami_qa_studies.bench_magami_qa_studies(seed))


def bench_mahagiri_qa_studies_family(seed: int = _SEED + 1):
    """mahagiri_qa_studies: synthetic correctness bench."""
    return _finite_blob(mahagiri_qa_studies.bench_mahagiri_qa_studies(seed))


def bench_min_kyawzwa_qa_studies_family(seed: int = _SEED + 2):
    """min_kyawzwa_qa_studies: synthetic correctness bench."""
    return _finite_blob(min_kyawzwa_qa_studies.bench_min_kyawzwa_qa_studies(seed))


def bench_shwe_nabay_qa_studies_family(seed: int = _SEED + 3):
    """shwe_nabay_qa_studies: synthetic correctness bench."""
    return _finite_blob(shwe_nabay_qa_studies.bench_shwe_nabay_qa_studies(seed))


def bench_taungmagyi_qa_studies_family(seed: int = _SEED + 4):
    """taungmagyi_qa_studies: synthetic correctness bench."""
    return _finite_blob(taungmagyi_qa_studies.bench_taungmagyi_qa_studies(seed))


def bench_thagya_min_qa_studies_family(seed: int = _SEED + 5):
    """thagya_min_qa_studies: synthetic correctness bench."""
    return _finite_blob(thagya_min_qa_studies.bench_thagya_min_qa_studies(seed))
