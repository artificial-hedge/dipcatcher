"""Wave-1406 bench adapters: temporal-QA canon (SYNTHETIC only)."""

from quant_fund.models import (
    menat_qa_studies,
    syndq_lite_studies,
    teas_qa_studies,
    time_qa_studies,
    timedial_qa_studies,
    timetravel_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14060


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_menat_qa_studies_family(seed: int = _SEED + 0):
    """menat_qa_studies: synthetic correctness bench."""
    return _finite_blob(menat_qa_studies.bench_menat_qa_studies(seed))


def bench_syndq_lite_studies_family(seed: int = _SEED + 1):
    """syndq_lite_studies: synthetic correctness bench."""
    return _finite_blob(syndq_lite_studies.bench_syndq_lite_studies(seed))


def bench_teas_qa_studies_family(seed: int = _SEED + 2):
    """teas_qa_studies: synthetic correctness bench."""
    return _finite_blob(teas_qa_studies.bench_teas_qa_studies(seed))


def bench_time_qa_studies_family(seed: int = _SEED + 3):
    """time_qa_studies: synthetic correctness bench."""
    return _finite_blob(time_qa_studies.bench_time_qa_studies(seed))


def bench_timedial_qa_studies_family(seed: int = _SEED + 4):
    """timedial_qa_studies: synthetic correctness bench."""
    return _finite_blob(timedial_qa_studies.bench_timedial_qa_studies(seed))


def bench_timetravel_lite_studies_family(seed: int = _SEED + 5):
    """timetravel_lite_studies: synthetic correctness bench."""
    return _finite_blob(timetravel_lite_studies.bench_timetravel_lite_studies(seed))
