"""Wave-1421 bench adapters: spatial-navigation canon (SYNTHETIC only)."""

from quant_fund.models import (
    geospatial_qa_studies,
    itinerary_qa_studies,
    journey_qa_studies,
    route_qa_studies,
    spatial_qa_studies,
    terrain_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14210


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


def bench_geospatial_qa_studies_family(seed: int = _SEED + 0):
    """geospatial_qa_studies: synthetic correctness bench."""
    return _finite_blob(geospatial_qa_studies.bench_geospatial_qa_studies(seed))


def bench_itinerary_qa_studies_family(seed: int = _SEED + 1):
    """itinerary_qa_studies: synthetic correctness bench."""
    return _finite_blob(itinerary_qa_studies.bench_itinerary_qa_studies(seed))


def bench_journey_qa_studies_family(seed: int = _SEED + 2):
    """journey_qa_studies: synthetic correctness bench."""
    return _finite_blob(journey_qa_studies.bench_journey_qa_studies(seed))


def bench_route_qa_studies_family(seed: int = _SEED + 3):
    """route_qa_studies: synthetic correctness bench."""
    return _finite_blob(route_qa_studies.bench_route_qa_studies(seed))


def bench_spatial_qa_studies_family(seed: int = _SEED + 4):
    """spatial_qa_studies: synthetic correctness bench."""
    return _finite_blob(spatial_qa_studies.bench_spatial_qa_studies(seed))


def bench_terrain_qa_studies_family(seed: int = _SEED + 5):
    """terrain_qa_studies: synthetic correctness bench."""
    return _finite_blob(terrain_qa_studies.bench_terrain_qa_studies(seed))
