"""Wave-1674 bench adapters: roman-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    genii_qa_studies,
    lares_qa_studies,
    larvae_qa_studies,
    lemures_qa_studies,
    manes_qa_studies,
    penates_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16740


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


def bench_genii_qa_studies_family(seed: int = _SEED + 0):
    """genii_qa_studies: synthetic correctness bench."""
    return _finite_blob(genii_qa_studies.bench_genii_qa_studies(seed))


def bench_lares_qa_studies_family(seed: int = _SEED + 1):
    """lares_qa_studies: synthetic correctness bench."""
    return _finite_blob(lares_qa_studies.bench_lares_qa_studies(seed))


def bench_larvae_qa_studies_family(seed: int = _SEED + 2):
    """larvae_qa_studies: synthetic correctness bench."""
    return _finite_blob(larvae_qa_studies.bench_larvae_qa_studies(seed))


def bench_lemures_qa_studies_family(seed: int = _SEED + 3):
    """lemures_qa_studies: synthetic correctness bench."""
    return _finite_blob(lemures_qa_studies.bench_lemures_qa_studies(seed))


def bench_manes_qa_studies_family(seed: int = _SEED + 4):
    """manes_qa_studies: synthetic correctness bench."""
    return _finite_blob(manes_qa_studies.bench_manes_qa_studies(seed))


def bench_penates_qa_studies_family(seed: int = _SEED + 5):
    """penates_qa_studies: synthetic correctness bench."""
    return _finite_blob(penates_qa_studies.bench_penates_qa_studies(seed))
