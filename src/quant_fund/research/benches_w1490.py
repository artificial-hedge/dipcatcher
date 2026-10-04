"""Wave-1490 bench adapters: blossom canon (SYNTHETIC only)."""

from quant_fund.models import (
    camellia_qa_studies,
    dahlia_qa_studies,
    sage_qa_studies,
    thyme_qa_studies,
    violet_qa_studies,
    zinnia_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14900


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_camellia_qa_studies_family(seed: int = _SEED + 0):
    """camellia_qa_studies: synthetic correctness bench."""
    return _finite_blob(camellia_qa_studies.bench_camellia_qa_studies(seed))


def bench_dahlia_qa_studies_family(seed: int = _SEED + 1):
    """dahlia_qa_studies: synthetic correctness bench."""
    return _finite_blob(dahlia_qa_studies.bench_dahlia_qa_studies(seed))


def bench_sage_qa_studies_family(seed: int = _SEED + 2):
    """sage_qa_studies: synthetic correctness bench."""
    return _finite_blob(sage_qa_studies.bench_sage_qa_studies(seed))


def bench_thyme_qa_studies_family(seed: int = _SEED + 3):
    """thyme_qa_studies: synthetic correctness bench."""
    return _finite_blob(thyme_qa_studies.bench_thyme_qa_studies(seed))


def bench_violet_qa_studies_family(seed: int = _SEED + 4):
    """violet_qa_studies: synthetic correctness bench."""
    return _finite_blob(violet_qa_studies.bench_violet_qa_studies(seed))


def bench_zinnia_qa_studies_family(seed: int = _SEED + 5):
    """zinnia_qa_studies: synthetic correctness bench."""
    return _finite_blob(zinnia_qa_studies.bench_zinnia_qa_studies(seed))
