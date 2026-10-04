"""Wave-1539 bench adapters: heron canon (SYNTHETIC only)."""

from quant_fund.models import (
    goliath_heron_qa_studies,
    green_heron_qa_studies,
    grey_heron_qa_studies,
    night_heron_qa_studies,
    purple_heron_qa_studies,
    tiger_heron_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15390


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_goliath_heron_qa_studies_family(seed: int = _SEED + 0):
    """goliath_heron_qa_studies: synthetic correctness bench."""
    return _finite_blob(goliath_heron_qa_studies.bench_goliath_heron_qa_studies(seed))


def bench_green_heron_qa_studies_family(seed: int = _SEED + 1):
    """green_heron_qa_studies: synthetic correctness bench."""
    return _finite_blob(green_heron_qa_studies.bench_green_heron_qa_studies(seed))


def bench_grey_heron_qa_studies_family(seed: int = _SEED + 2):
    """grey_heron_qa_studies: synthetic correctness bench."""
    return _finite_blob(grey_heron_qa_studies.bench_grey_heron_qa_studies(seed))


def bench_night_heron_qa_studies_family(seed: int = _SEED + 3):
    """night_heron_qa_studies: synthetic correctness bench."""
    return _finite_blob(night_heron_qa_studies.bench_night_heron_qa_studies(seed))


def bench_purple_heron_qa_studies_family(seed: int = _SEED + 4):
    """purple_heron_qa_studies: synthetic correctness bench."""
    return _finite_blob(purple_heron_qa_studies.bench_purple_heron_qa_studies(seed))


def bench_tiger_heron_qa_studies_family(seed: int = _SEED + 5):
    """tiger_heron_qa_studies: synthetic correctness bench."""
    return _finite_blob(tiger_heron_qa_studies.bench_tiger_heron_qa_studies(seed))
