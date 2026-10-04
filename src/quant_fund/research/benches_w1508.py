"""Wave-1508 bench adapters: songbird canon (SYNTHETIC only)."""

from quant_fund.models import (
    chickadee_qa_studies,
    finch_qa_studies,
    sparrow_qa_studies,
    thrush_qa_studies,
    warbler_qa_studies,
    wren_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15080


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_chickadee_qa_studies_family(seed: int = _SEED + 0):
    """chickadee_qa_studies: synthetic correctness bench."""
    return _finite_blob(chickadee_qa_studies.bench_chickadee_qa_studies(seed))


def bench_finch_qa_studies_family(seed: int = _SEED + 1):
    """finch_qa_studies: synthetic correctness bench."""
    return _finite_blob(finch_qa_studies.bench_finch_qa_studies(seed))


def bench_sparrow_qa_studies_family(seed: int = _SEED + 2):
    """sparrow_qa_studies: synthetic correctness bench."""
    return _finite_blob(sparrow_qa_studies.bench_sparrow_qa_studies(seed))


def bench_thrush_qa_studies_family(seed: int = _SEED + 3):
    """thrush_qa_studies: synthetic correctness bench."""
    return _finite_blob(thrush_qa_studies.bench_thrush_qa_studies(seed))


def bench_warbler_qa_studies_family(seed: int = _SEED + 4):
    """warbler_qa_studies: synthetic correctness bench."""
    return _finite_blob(warbler_qa_studies.bench_warbler_qa_studies(seed))


def bench_wren_qa_studies_family(seed: int = _SEED + 5):
    """wren_qa_studies: synthetic correctness bench."""
    return _finite_blob(wren_qa_studies.bench_wren_qa_studies(seed))
