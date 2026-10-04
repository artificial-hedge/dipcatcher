"""Wave-1308 bench adapters: long-context-factuality canon (SYNTHETIC only)."""

from quant_fund.models import (
    halu_eval_studies,
    infinite_bench_studies,
    longmem_studies,
    needle_haystack_studies,
    ruler_studies,
    truthful_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13080


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_halu_eval_studies_family(seed: int = _SEED + 0):
    """halu_eval_studies: synthetic correctness bench."""
    return _finite_blob(halu_eval_studies.bench_halu_eval_studies(seed))


def bench_infinite_bench_studies_family(seed: int = _SEED + 1):
    """infinite_bench_studies: synthetic correctness bench."""
    return _finite_blob(infinite_bench_studies.bench_infinite_bench_studies(seed))


def bench_longmem_studies_family(seed: int = _SEED + 2):
    """longmem_studies: synthetic correctness bench."""
    return _finite_blob(longmem_studies.bench_longmem_studies(seed))


def bench_needle_haystack_studies_family(seed: int = _SEED + 3):
    """needle_haystack_studies: synthetic correctness bench."""
    return _finite_blob(needle_haystack_studies.bench_needle_haystack_studies(seed))


def bench_ruler_studies_family(seed: int = _SEED + 4):
    """ruler_studies: synthetic correctness bench."""
    return _finite_blob(ruler_studies.bench_ruler_studies(seed))


def bench_truthful_qa_studies_family(seed: int = _SEED + 5):
    """truthful_qa_studies: synthetic correctness bench."""
    return _finite_blob(truthful_qa_studies.bench_truthful_qa_studies(seed))
