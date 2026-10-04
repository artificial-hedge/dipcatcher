"""Wave-1474 bench adapters: waterbird canon (SYNTHETIC only)."""

from quant_fund.models import (
    bittern_qa_studies,
    cormorant_qa_studies,
    curlew_qa_studies,
    ibis_qa_studies,
    kingfisher_qa_studies,
    loon_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14740


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bittern_qa_studies_family(seed: int = _SEED + 0):
    """bittern_qa_studies: synthetic correctness bench."""
    return _finite_blob(bittern_qa_studies.bench_bittern_qa_studies(seed))


def bench_cormorant_qa_studies_family(seed: int = _SEED + 1):
    """cormorant_qa_studies: synthetic correctness bench."""
    return _finite_blob(cormorant_qa_studies.bench_cormorant_qa_studies(seed))


def bench_curlew_qa_studies_family(seed: int = _SEED + 2):
    """curlew_qa_studies: synthetic correctness bench."""
    return _finite_blob(curlew_qa_studies.bench_curlew_qa_studies(seed))


def bench_ibis_qa_studies_family(seed: int = _SEED + 3):
    """ibis_qa_studies: synthetic correctness bench."""
    return _finite_blob(ibis_qa_studies.bench_ibis_qa_studies(seed))


def bench_kingfisher_qa_studies_family(seed: int = _SEED + 4):
    """kingfisher_qa_studies: synthetic correctness bench."""
    return _finite_blob(kingfisher_qa_studies.bench_kingfisher_qa_studies(seed))


def bench_loon_qa_studies_family(seed: int = _SEED + 5):
    """loon_qa_studies: synthetic correctness bench."""
    return _finite_blob(loon_qa_studies.bench_loon_qa_studies(seed))
