"""Wave-1498 bench adapters: seabird-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    auklet_qa_studies,
    booby_qa_studies,
    frigatebird_qa_studies,
    guillemot_qa_studies,
    murrelet_qa_studies,
    razorbill_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14980


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


def bench_auklet_qa_studies_family(seed: int = _SEED + 0):
    """auklet_qa_studies: synthetic correctness bench."""
    return _finite_blob(auklet_qa_studies.bench_auklet_qa_studies(seed))


def bench_booby_qa_studies_family(seed: int = _SEED + 1):
    """booby_qa_studies: synthetic correctness bench."""
    return _finite_blob(booby_qa_studies.bench_booby_qa_studies(seed))


def bench_frigatebird_qa_studies_family(seed: int = _SEED + 2):
    """frigatebird_qa_studies: synthetic correctness bench."""
    return _finite_blob(frigatebird_qa_studies.bench_frigatebird_qa_studies(seed))


def bench_guillemot_qa_studies_family(seed: int = _SEED + 3):
    """guillemot_qa_studies: synthetic correctness bench."""
    return _finite_blob(guillemot_qa_studies.bench_guillemot_qa_studies(seed))


def bench_murrelet_qa_studies_family(seed: int = _SEED + 4):
    """murrelet_qa_studies: synthetic correctness bench."""
    return _finite_blob(murrelet_qa_studies.bench_murrelet_qa_studies(seed))


def bench_razorbill_qa_studies_family(seed: int = _SEED + 5):
    """razorbill_qa_studies: synthetic correctness bench."""
    return _finite_blob(razorbill_qa_studies.bench_razorbill_qa_studies(seed))
