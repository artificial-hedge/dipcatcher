"""Wave-1512 bench adapters: shorebird-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    dunlin_qa_studies,
    knot_qa_studies,
    oystercatcher_qa_studies,
    phalarope_qa_studies,
    stilt_qa_studies,
    whimbrel_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15120


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


def bench_dunlin_qa_studies_family(seed: int = _SEED + 0):
    """dunlin_qa_studies: synthetic correctness bench."""
    return _finite_blob(dunlin_qa_studies.bench_dunlin_qa_studies(seed))


def bench_knot_qa_studies_family(seed: int = _SEED + 1):
    """knot_qa_studies: synthetic correctness bench."""
    return _finite_blob(knot_qa_studies.bench_knot_qa_studies(seed))


def bench_oystercatcher_qa_studies_family(seed: int = _SEED + 2):
    """oystercatcher_qa_studies: synthetic correctness bench."""
    return _finite_blob(oystercatcher_qa_studies.bench_oystercatcher_qa_studies(seed))


def bench_phalarope_qa_studies_family(seed: int = _SEED + 3):
    """phalarope_qa_studies: synthetic correctness bench."""
    return _finite_blob(phalarope_qa_studies.bench_phalarope_qa_studies(seed))


def bench_stilt_qa_studies_family(seed: int = _SEED + 4):
    """stilt_qa_studies: synthetic correctness bench."""
    return _finite_blob(stilt_qa_studies.bench_stilt_qa_studies(seed))


def bench_whimbrel_qa_studies_family(seed: int = _SEED + 5):
    """whimbrel_qa_studies: synthetic correctness bench."""
    return _finite_blob(whimbrel_qa_studies.bench_whimbrel_qa_studies(seed))
