"""Wave-1633 bench adapters: elemental canon (SYNTHETIC only)."""

from quant_fund.models import (
    air_sylph_qa_studies,
    earth_golem_qa_studies,
    fire_spirit_qa_studies,
    frost_wight_qa_studies,
    storm_jinn_qa_studies,
    water_sprite_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16330


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


def bench_air_sylph_qa_studies_family(seed: int = _SEED + 0):
    """air_sylph_qa_studies: synthetic correctness bench."""
    return _finite_blob(air_sylph_qa_studies.bench_air_sylph_qa_studies(seed))


def bench_earth_golem_qa_studies_family(seed: int = _SEED + 1):
    """earth_golem_qa_studies: synthetic correctness bench."""
    return _finite_blob(earth_golem_qa_studies.bench_earth_golem_qa_studies(seed))


def bench_fire_spirit_qa_studies_family(seed: int = _SEED + 2):
    """fire_spirit_qa_studies: synthetic correctness bench."""
    return _finite_blob(fire_spirit_qa_studies.bench_fire_spirit_qa_studies(seed))


def bench_frost_wight_qa_studies_family(seed: int = _SEED + 3):
    """frost_wight_qa_studies: synthetic correctness bench."""
    return _finite_blob(frost_wight_qa_studies.bench_frost_wight_qa_studies(seed))


def bench_storm_jinn_qa_studies_family(seed: int = _SEED + 4):
    """storm_jinn_qa_studies: synthetic correctness bench."""
    return _finite_blob(storm_jinn_qa_studies.bench_storm_jinn_qa_studies(seed))


def bench_water_sprite_qa_studies_family(seed: int = _SEED + 5):
    """water_sprite_qa_studies: synthetic correctness bench."""
    return _finite_blob(water_sprite_qa_studies.bench_water_sprite_qa_studies(seed))
