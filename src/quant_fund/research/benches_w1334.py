"""Wave-1334 bench adapters: long-context-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    books_qa_studies,
    lcc_codebase_studies,
    multi_news_eval_studies,
    narrative_qa_studies,
    needle_multi_studies,
    qmsum_eval_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13340


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_books_qa_studies_family(seed: int = _SEED + 0):
    """books_qa_studies: synthetic correctness bench."""
    return _finite_blob(books_qa_studies.bench_books_qa_studies(seed))


def bench_lcc_codebase_studies_family(seed: int = _SEED + 1):
    """lcc_codebase_studies: synthetic correctness bench."""
    return _finite_blob(lcc_codebase_studies.bench_lcc_codebase_studies(seed))


def bench_multi_news_eval_studies_family(seed: int = _SEED + 2):
    """multi_news_eval_studies: synthetic correctness bench."""
    return _finite_blob(multi_news_eval_studies.bench_multi_news_eval_studies(seed))


def bench_narrative_qa_studies_family(seed: int = _SEED + 3):
    """narrative_qa_studies: synthetic correctness bench."""
    return _finite_blob(narrative_qa_studies.bench_narrative_qa_studies(seed))


def bench_needle_multi_studies_family(seed: int = _SEED + 4):
    """needle_multi_studies: synthetic correctness bench."""
    return _finite_blob(needle_multi_studies.bench_needle_multi_studies(seed))


def bench_qmsum_eval_studies_family(seed: int = _SEED + 5):
    """qmsum_eval_studies: synthetic correctness bench."""
    return _finite_blob(qmsum_eval_studies.bench_qmsum_eval_studies(seed))
