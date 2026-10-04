"""Wave-1727 bench adapters: mongolian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    erlug_qa_studies,
    etseg_qa_studies,
    manzan_qa_studies,
    otgon_qa_studies,
    tenger_qa_studies,
    ulgan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17270


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_erlug_qa_studies_family(seed: int = _SEED + 0):
    """erlug_qa_studies: synthetic correctness bench."""
    return _finite_blob(erlug_qa_studies.bench_erlug_qa_studies(seed))


def bench_etseg_qa_studies_family(seed: int = _SEED + 1):
    """etseg_qa_studies: synthetic correctness bench."""
    return _finite_blob(etseg_qa_studies.bench_etseg_qa_studies(seed))


def bench_manzan_qa_studies_family(seed: int = _SEED + 2):
    """manzan_qa_studies: synthetic correctness bench."""
    return _finite_blob(manzan_qa_studies.bench_manzan_qa_studies(seed))


def bench_otgon_qa_studies_family(seed: int = _SEED + 3):
    """otgon_qa_studies: synthetic correctness bench."""
    return _finite_blob(otgon_qa_studies.bench_otgon_qa_studies(seed))


def bench_tenger_qa_studies_family(seed: int = _SEED + 4):
    """tenger_qa_studies: synthetic correctness bench."""
    return _finite_blob(tenger_qa_studies.bench_tenger_qa_studies(seed))


def bench_ulgan_qa_studies_family(seed: int = _SEED + 5):
    """ulgan_qa_studies: synthetic correctness bench."""
    return _finite_blob(ulgan_qa_studies.bench_ulgan_qa_studies(seed))
