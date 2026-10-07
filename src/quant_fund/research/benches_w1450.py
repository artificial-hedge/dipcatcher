"""Wave-1450 bench adapters: farm canon (SYNTHETIC only)."""

from quant_fund.models import (
    barn_qa_studies,
    cow_qa_studies,
    goat_qa_studies,
    horse_qa_studies,
    pig_qa_studies,
    sheep_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14500


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


def bench_barn_qa_studies_family(seed: int = _SEED + 0):
    """barn_qa_studies: synthetic correctness bench."""
    return _finite_blob(barn_qa_studies.bench_barn_qa_studies(seed))


def bench_cow_qa_studies_family(seed: int = _SEED + 1):
    """cow_qa_studies: synthetic correctness bench."""
    return _finite_blob(cow_qa_studies.bench_cow_qa_studies(seed))


def bench_goat_qa_studies_family(seed: int = _SEED + 2):
    """goat_qa_studies: synthetic correctness bench."""
    return _finite_blob(goat_qa_studies.bench_goat_qa_studies(seed))


def bench_horse_qa_studies_family(seed: int = _SEED + 3):
    """horse_qa_studies: synthetic correctness bench."""
    return _finite_blob(horse_qa_studies.bench_horse_qa_studies(seed))


def bench_pig_qa_studies_family(seed: int = _SEED + 4):
    """pig_qa_studies: synthetic correctness bench."""
    return _finite_blob(pig_qa_studies.bench_pig_qa_studies(seed))


def bench_sheep_qa_studies_family(seed: int = _SEED + 5):
    """sheep_qa_studies: synthetic correctness bench."""
    return _finite_blob(sheep_qa_studies.bench_sheep_qa_studies(seed))
