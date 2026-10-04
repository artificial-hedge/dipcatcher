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
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
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
