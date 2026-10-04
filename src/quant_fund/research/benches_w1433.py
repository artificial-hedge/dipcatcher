"""Wave-1433 bench adapters: weather canon (SYNTHETIC only)."""

from quant_fund.models import (
    cloud_qa_studies,
    frost_qa_studies,
    hurricane_qa_studies,
    rain_qa_studies,
    storm_qa_studies,
    wind_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14330


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cloud_qa_studies_family(seed: int = _SEED + 0):
    """cloud_qa_studies: synthetic correctness bench."""
    return _finite_blob(cloud_qa_studies.bench_cloud_qa_studies(seed))


def bench_frost_qa_studies_family(seed: int = _SEED + 1):
    """frost_qa_studies: synthetic correctness bench."""
    return _finite_blob(frost_qa_studies.bench_frost_qa_studies(seed))


def bench_hurricane_qa_studies_family(seed: int = _SEED + 2):
    """hurricane_qa_studies: synthetic correctness bench."""
    return _finite_blob(hurricane_qa_studies.bench_hurricane_qa_studies(seed))


def bench_rain_qa_studies_family(seed: int = _SEED + 3):
    """rain_qa_studies: synthetic correctness bench."""
    return _finite_blob(rain_qa_studies.bench_rain_qa_studies(seed))


def bench_storm_qa_studies_family(seed: int = _SEED + 4):
    """storm_qa_studies: synthetic correctness bench."""
    return _finite_blob(storm_qa_studies.bench_storm_qa_studies(seed))


def bench_wind_qa_studies_family(seed: int = _SEED + 5):
    """wind_qa_studies: synthetic correctness bench."""
    return _finite_blob(wind_qa_studies.bench_wind_qa_studies(seed))
