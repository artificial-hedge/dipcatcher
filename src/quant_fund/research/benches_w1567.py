"""Wave-1567 bench adapters: cyprinid canon (SYNTHETIC only)."""

from quant_fund.models import (
    barbel_qa_studies,
    bream_qa_studies,
    carp_qa_studies,
    minnow_qa_studies,
    roach_qa_studies,
    tench_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15670


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_barbel_qa_studies_family(seed: int = _SEED + 0):
    """barbel_qa_studies: synthetic correctness bench."""
    return _finite_blob(barbel_qa_studies.bench_barbel_qa_studies(seed))


def bench_bream_qa_studies_family(seed: int = _SEED + 1):
    """bream_qa_studies: synthetic correctness bench."""
    return _finite_blob(bream_qa_studies.bench_bream_qa_studies(seed))


def bench_carp_qa_studies_family(seed: int = _SEED + 2):
    """carp_qa_studies: synthetic correctness bench."""
    return _finite_blob(carp_qa_studies.bench_carp_qa_studies(seed))


def bench_minnow_qa_studies_family(seed: int = _SEED + 3):
    """minnow_qa_studies: synthetic correctness bench."""
    return _finite_blob(minnow_qa_studies.bench_minnow_qa_studies(seed))


def bench_roach_qa_studies_family(seed: int = _SEED + 4):
    """roach_qa_studies: synthetic correctness bench."""
    return _finite_blob(roach_qa_studies.bench_roach_qa_studies(seed))


def bench_tench_qa_studies_family(seed: int = _SEED + 5):
    """tench_qa_studies: synthetic correctness bench."""
    return _finite_blob(tench_qa_studies.bench_tench_qa_studies(seed))
