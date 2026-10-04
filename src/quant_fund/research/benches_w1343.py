"""Wave-1343 bench adapters: reading-comprehension-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    coqa_qa_studies,
    drop_qa_studies,
    news_qa_studies,
    quac_qa_studies,
    quail_qa_studies,
    quoref_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13430


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_coqa_qa_studies_family(seed: int = _SEED + 0):
    """coqa_qa_studies: synthetic correctness bench."""
    return _finite_blob(coqa_qa_studies.bench_coqa_qa_studies(seed))


def bench_drop_qa_studies_family(seed: int = _SEED + 1):
    """drop_qa_studies: synthetic correctness bench."""
    return _finite_blob(drop_qa_studies.bench_drop_qa_studies(seed))


def bench_news_qa_studies_family(seed: int = _SEED + 2):
    """news_qa_studies: synthetic correctness bench."""
    return _finite_blob(news_qa_studies.bench_news_qa_studies(seed))


def bench_quac_qa_studies_family(seed: int = _SEED + 3):
    """quac_qa_studies: synthetic correctness bench."""
    return _finite_blob(quac_qa_studies.bench_quac_qa_studies(seed))


def bench_quail_qa_studies_family(seed: int = _SEED + 4):
    """quail_qa_studies: synthetic correctness bench."""
    return _finite_blob(quail_qa_studies.bench_quail_qa_studies(seed))


def bench_quoref_qa_studies_family(seed: int = _SEED + 5):
    """quoref_qa_studies: synthetic correctness bench."""
    return _finite_blob(quoref_qa_studies.bench_quoref_qa_studies(seed))
