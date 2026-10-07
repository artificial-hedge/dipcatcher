"""Wave-1622 bench adapters: alpine-ridgeline canon (SYNTHETIC only)."""

from quant_fund.models import (
    barbary_qa_studies,
    blue_sheep_qa_studies,
    himalayan_qa_studies,
    nilgiri_qa_studies,
    snow_leopard_qa_studies,
    snowcock_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16220


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


def bench_snowcock_qa_studies_family(seed: int = _SEED + 0):
    """snowcock_qa_studies: synthetic correctness bench."""
    return _finite_blob(snowcock_qa_studies.bench_snowcock_qa_studies(seed))


def bench_barbary_qa_studies_family(seed: int = _SEED + 1):
    """barbary_qa_studies: synthetic correctness bench."""
    return _finite_blob(barbary_qa_studies.bench_barbary_qa_studies(seed))


def bench_blue_sheep_qa_studies_family(seed: int = _SEED + 2):
    """blue_sheep_qa_studies: synthetic correctness bench."""
    return _finite_blob(blue_sheep_qa_studies.bench_blue_sheep_qa_studies(seed))


def bench_himalayan_qa_studies_family(seed: int = _SEED + 3):
    """himalayan_qa_studies: synthetic correctness bench."""
    return _finite_blob(himalayan_qa_studies.bench_himalayan_qa_studies(seed))


def bench_nilgiri_qa_studies_family(seed: int = _SEED + 4):
    """nilgiri_qa_studies: synthetic correctness bench."""
    return _finite_blob(nilgiri_qa_studies.bench_nilgiri_qa_studies(seed))


def bench_snow_leopard_qa_studies_family(seed: int = _SEED + 5):
    """snow_leopard_qa_studies: synthetic correctness bench."""
    return _finite_blob(snow_leopard_qa_studies.bench_snow_leopard_qa_studies(seed))
