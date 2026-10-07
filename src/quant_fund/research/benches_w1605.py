"""Wave-1605 bench adapters: intertidal-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    decorator_qa_studies,
    fiddler_qa_studies,
    rock_crab_qa_studies,
    sea_snake_qa_studies,
    skate_qa_studies,
    wobbegong_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16050


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


def bench_decorator_qa_studies_family(seed: int = _SEED + 0):
    """decorator_qa_studies: synthetic correctness bench."""
    return _finite_blob(decorator_qa_studies.bench_decorator_qa_studies(seed))


def bench_fiddler_qa_studies_family(seed: int = _SEED + 1):
    """fiddler_qa_studies: synthetic correctness bench."""
    return _finite_blob(fiddler_qa_studies.bench_fiddler_qa_studies(seed))


def bench_rock_crab_qa_studies_family(seed: int = _SEED + 2):
    """rock_crab_qa_studies: synthetic correctness bench."""
    return _finite_blob(rock_crab_qa_studies.bench_rock_crab_qa_studies(seed))


def bench_sea_snake_qa_studies_family(seed: int = _SEED + 3):
    """sea_snake_qa_studies: synthetic correctness bench."""
    return _finite_blob(sea_snake_qa_studies.bench_sea_snake_qa_studies(seed))


def bench_skate_qa_studies_family(seed: int = _SEED + 4):
    """skate_qa_studies: synthetic correctness bench."""
    return _finite_blob(skate_qa_studies.bench_skate_qa_studies(seed))


def bench_wobbegong_qa_studies_family(seed: int = _SEED + 5):
    """wobbegong_qa_studies: synthetic correctness bench."""
    return _finite_blob(wobbegong_qa_studies.bench_wobbegong_qa_studies(seed))
