"""Wave-1425 bench adapters: media canon (SYNTHETIC only)."""

from quant_fund.models import (
    article_qa_studies,
    broadcast_qa_studies,
    column_qa_studies,
    debate_qa_studies,
    editorial_qa_studies,
    headline_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14250


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_article_qa_studies_family(seed: int = _SEED + 0):
    """article_qa_studies: synthetic correctness bench."""
    return _finite_blob(article_qa_studies.bench_article_qa_studies(seed))


def bench_broadcast_qa_studies_family(seed: int = _SEED + 1):
    """broadcast_qa_studies: synthetic correctness bench."""
    return _finite_blob(broadcast_qa_studies.bench_broadcast_qa_studies(seed))


def bench_column_qa_studies_family(seed: int = _SEED + 2):
    """column_qa_studies: synthetic correctness bench."""
    return _finite_blob(column_qa_studies.bench_column_qa_studies(seed))


def bench_debate_qa_studies_family(seed: int = _SEED + 3):
    """debate_qa_studies: synthetic correctness bench."""
    return _finite_blob(debate_qa_studies.bench_debate_qa_studies(seed))


def bench_editorial_qa_studies_family(seed: int = _SEED + 4):
    """editorial_qa_studies: synthetic correctness bench."""
    return _finite_blob(editorial_qa_studies.bench_editorial_qa_studies(seed))


def bench_headline_qa_studies_family(seed: int = _SEED + 5):
    """headline_qa_studies: synthetic correctness bench."""
    return _finite_blob(headline_qa_studies.bench_headline_qa_studies(seed))
