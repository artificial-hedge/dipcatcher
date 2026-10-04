"""Wave-1388 bench adapters: RAG-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    corpus_qa_studies,
    crag_bench_studies,
    domain_rag_studies,
    freshqa_studies,
    ragas_lite_studies,
    rgb_eval_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13880


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_corpus_qa_studies_family(seed: int = _SEED + 0):
    """corpus_qa_studies: synthetic correctness bench."""
    return _finite_blob(corpus_qa_studies.bench_corpus_qa_studies(seed))


def bench_crag_bench_studies_family(seed: int = _SEED + 1):
    """crag_bench_studies: synthetic correctness bench."""
    return _finite_blob(crag_bench_studies.bench_crag_bench_studies(seed))


def bench_domain_rag_studies_family(seed: int = _SEED + 2):
    """domain_rag_studies: synthetic correctness bench."""
    return _finite_blob(domain_rag_studies.bench_domain_rag_studies(seed))


def bench_freshqa_studies_family(seed: int = _SEED + 3):
    """freshqa_studies: synthetic correctness bench."""
    return _finite_blob(freshqa_studies.bench_freshqa_studies(seed))


def bench_ragas_lite_studies_family(seed: int = _SEED + 4):
    """ragas_lite_studies: synthetic correctness bench."""
    return _finite_blob(ragas_lite_studies.bench_ragas_lite_studies(seed))


def bench_rgb_eval_studies_family(seed: int = _SEED + 5):
    """rgb_eval_studies: synthetic correctness bench."""
    return _finite_blob(rgb_eval_studies.bench_rgb_eval_studies(seed))
