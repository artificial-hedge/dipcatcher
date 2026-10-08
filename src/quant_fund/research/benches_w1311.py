"""Wave-1311 bench adapters: safety-benchmark canon (SYNTHETIC only)."""

from quant_fund.models import (
    aegis_studies,
    air_bench_studies,
    overkill_studies,
    salad_bench_studies,
    sorry_bench_studies,
    wildguard_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13110


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


def bench_aegis_studies_family(seed: int = _SEED + 0):
    """aegis_studies: synthetic correctness bench."""
    return _finite_blob(aegis_studies.bench_aegis_studies(seed))


def bench_air_bench_studies_family(seed: int = _SEED + 1):
    """air_bench_studies: synthetic correctness bench."""
    return _finite_blob(air_bench_studies.bench_air_bench_studies(seed))


def bench_overkill_studies_family(seed: int = _SEED + 2):
    """overkill_studies: synthetic correctness bench."""
    return _finite_blob(overkill_studies.bench_overkill_studies(seed))


def bench_salad_bench_studies_family(seed: int = _SEED + 3):
    """salad_bench_studies: synthetic correctness bench."""
    return _finite_blob(salad_bench_studies.bench_salad_bench_studies(seed))


def bench_sorry_bench_studies_family(seed: int = _SEED + 4):
    """sorry_bench_studies: synthetic correctness bench."""
    return _finite_blob(sorry_bench_studies.bench_sorry_bench_studies(seed))


def bench_wildguard_studies_family(seed: int = _SEED + 5):
    """wildguard_studies: synthetic correctness bench."""
    return _finite_blob(wildguard_studies.bench_wildguard_studies(seed))
