"""Wave-1436 bench adapters: celestial canon (SYNTHETIC only)."""

from quant_fund.models import (
    comet_qa_studies,
    galaxy_qa_studies,
    moon_qa_studies,
    nebula_qa_studies,
    planet_qa_studies,
    star_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14360


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


def bench_comet_qa_studies_family(seed: int = _SEED + 0):
    """comet_qa_studies: synthetic correctness bench."""
    return _finite_blob(comet_qa_studies.bench_comet_qa_studies(seed))


def bench_galaxy_qa_studies_family(seed: int = _SEED + 1):
    """galaxy_qa_studies: synthetic correctness bench."""
    return _finite_blob(galaxy_qa_studies.bench_galaxy_qa_studies(seed))


def bench_moon_qa_studies_family(seed: int = _SEED + 2):
    """moon_qa_studies: synthetic correctness bench."""
    return _finite_blob(moon_qa_studies.bench_moon_qa_studies(seed))


def bench_nebula_qa_studies_family(seed: int = _SEED + 3):
    """nebula_qa_studies: synthetic correctness bench."""
    return _finite_blob(nebula_qa_studies.bench_nebula_qa_studies(seed))


def bench_planet_qa_studies_family(seed: int = _SEED + 4):
    """planet_qa_studies: synthetic correctness bench."""
    return _finite_blob(planet_qa_studies.bench_planet_qa_studies(seed))


def bench_star_qa_studies_family(seed: int = _SEED + 5):
    """star_qa_studies: synthetic correctness bench."""
    return _finite_blob(star_qa_studies.bench_star_qa_studies(seed))
