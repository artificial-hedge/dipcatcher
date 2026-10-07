"""Wave-1389 bench adapters: long-doc-sum canon (SYNTHETIC only)."""

from quant_fund.models import (
    book_sum_studies,
    fanout_qa_studies,
    infinitesum_studies,
    marlense_studies,
    narra_sum_studies,
    quote_sum_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13890


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_book_sum_studies_family(seed: int = _SEED + 0):
    """book_sum_studies: synthetic correctness bench."""
    return _finite_blob(book_sum_studies.bench_book_sum_studies(seed))


def bench_fanout_qa_studies_family(seed: int = _SEED + 1):
    """fanout_qa_studies: synthetic correctness bench."""
    return _finite_blob(fanout_qa_studies.bench_fanout_qa_studies(seed))


def bench_infinitesum_studies_family(seed: int = _SEED + 2):
    """infinitesum_studies: synthetic correctness bench."""
    return _finite_blob(infinitesum_studies.bench_infinitesum_studies(seed))


def bench_marlense_studies_family(seed: int = _SEED + 3):
    """marlense_studies: synthetic correctness bench."""
    return _finite_blob(marlense_studies.bench_marlense_studies(seed))


def bench_narra_sum_studies_family(seed: int = _SEED + 4):
    """narra_sum_studies: synthetic correctness bench."""
    return _finite_blob(narra_sum_studies.bench_narra_sum_studies(seed))


def bench_quote_sum_studies_family(seed: int = _SEED + 5):
    """quote_sum_studies: synthetic correctness bench."""
    return _finite_blob(quote_sum_studies.bench_quote_sum_studies(seed))
