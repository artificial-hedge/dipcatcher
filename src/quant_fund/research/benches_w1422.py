"""Wave-1422 bench adapters: temporal-era canon (SYNTHETIC only)."""

from quant_fund.models import (
    calendar_qa_studies,
    century_qa_studies,
    date_qa_studies,
    decade_qa_studies,
    epoch_qa_studies,
    era_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14220


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_calendar_qa_studies_family(seed: int = _SEED + 0):
    """calendar_qa_studies: synthetic correctness bench."""
    return _finite_blob(calendar_qa_studies.bench_calendar_qa_studies(seed))


def bench_century_qa_studies_family(seed: int = _SEED + 1):
    """century_qa_studies: synthetic correctness bench."""
    return _finite_blob(century_qa_studies.bench_century_qa_studies(seed))


def bench_date_qa_studies_family(seed: int = _SEED + 2):
    """date_qa_studies: synthetic correctness bench."""
    return _finite_blob(date_qa_studies.bench_date_qa_studies(seed))


def bench_decade_qa_studies_family(seed: int = _SEED + 3):
    """decade_qa_studies: synthetic correctness bench."""
    return _finite_blob(decade_qa_studies.bench_decade_qa_studies(seed))


def bench_epoch_qa_studies_family(seed: int = _SEED + 4):
    """epoch_qa_studies: synthetic correctness bench."""
    return _finite_blob(epoch_qa_studies.bench_epoch_qa_studies(seed))


def bench_era_qa_studies_family(seed: int = _SEED + 5):
    """era_qa_studies: synthetic correctness bench."""
    return _finite_blob(era_qa_studies.bench_era_qa_studies(seed))
