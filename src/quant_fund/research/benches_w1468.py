"""Wave-1468 bench adapters: moorland canon (SYNTHETIC only)."""

from quant_fund.models import (
    dale_qa_studies,
    fen_qa_studies,
    glen_qa_studies,
    heath_qa_studies,
    knoll_qa_studies,
    moor_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14680


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


def bench_dale_qa_studies_family(seed: int = _SEED + 0):
    """dale_qa_studies: synthetic correctness bench."""
    return _finite_blob(dale_qa_studies.bench_dale_qa_studies(seed))


def bench_fen_qa_studies_family(seed: int = _SEED + 1):
    """fen_qa_studies: synthetic correctness bench."""
    return _finite_blob(fen_qa_studies.bench_fen_qa_studies(seed))


def bench_glen_qa_studies_family(seed: int = _SEED + 2):
    """glen_qa_studies: synthetic correctness bench."""
    return _finite_blob(glen_qa_studies.bench_glen_qa_studies(seed))


def bench_heath_qa_studies_family(seed: int = _SEED + 3):
    """heath_qa_studies: synthetic correctness bench."""
    return _finite_blob(heath_qa_studies.bench_heath_qa_studies(seed))


def bench_knoll_qa_studies_family(seed: int = _SEED + 4):
    """knoll_qa_studies: synthetic correctness bench."""
    return _finite_blob(knoll_qa_studies.bench_knoll_qa_studies(seed))


def bench_moor_qa_studies_family(seed: int = _SEED + 5):
    """moor_qa_studies: synthetic correctness bench."""
    return _finite_blob(moor_qa_studies.bench_moor_qa_studies(seed))
