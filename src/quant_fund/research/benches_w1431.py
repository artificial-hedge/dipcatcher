"""Wave-1431 bench adapters: vehicle canon (SYNTHETIC only)."""

from quant_fund.models import (
    aircraft_qa_studies,
    bike_qa_studies,
    bus_qa_studies,
    car_qa_studies,
    engine_qa_studies,
    plane_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14310


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aircraft_qa_studies_family(seed: int = _SEED + 0):
    """aircraft_qa_studies: synthetic correctness bench."""
    return _finite_blob(aircraft_qa_studies.bench_aircraft_qa_studies(seed))


def bench_bike_qa_studies_family(seed: int = _SEED + 1):
    """bike_qa_studies: synthetic correctness bench."""
    return _finite_blob(bike_qa_studies.bench_bike_qa_studies(seed))


def bench_bus_qa_studies_family(seed: int = _SEED + 2):
    """bus_qa_studies: synthetic correctness bench."""
    return _finite_blob(bus_qa_studies.bench_bus_qa_studies(seed))


def bench_car_qa_studies_family(seed: int = _SEED + 3):
    """car_qa_studies: synthetic correctness bench."""
    return _finite_blob(car_qa_studies.bench_car_qa_studies(seed))


def bench_engine_qa_studies_family(seed: int = _SEED + 4):
    """engine_qa_studies: synthetic correctness bench."""
    return _finite_blob(engine_qa_studies.bench_engine_qa_studies(seed))


def bench_plane_qa_studies_family(seed: int = _SEED + 5):
    """plane_qa_studies: synthetic correctness bench."""
    return _finite_blob(plane_qa_studies.bench_plane_qa_studies(seed))
