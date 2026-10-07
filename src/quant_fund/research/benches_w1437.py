"""Wave-1437 bench adapters: mythic canon (SYNTHETIC only)."""

from quant_fund.models import (
    deity_qa_studies,
    dragon_qa_studies,
    hero_qa_studies,
    olympus_qa_studies,
    phoenix_qa_studies,
    titan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14370


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


def bench_deity_qa_studies_family(seed: int = _SEED + 0):
    """deity_qa_studies: synthetic correctness bench."""
    return _finite_blob(deity_qa_studies.bench_deity_qa_studies(seed))


def bench_dragon_qa_studies_family(seed: int = _SEED + 1):
    """dragon_qa_studies: synthetic correctness bench."""
    return _finite_blob(dragon_qa_studies.bench_dragon_qa_studies(seed))


def bench_hero_qa_studies_family(seed: int = _SEED + 2):
    """hero_qa_studies: synthetic correctness bench."""
    return _finite_blob(hero_qa_studies.bench_hero_qa_studies(seed))


def bench_olympus_qa_studies_family(seed: int = _SEED + 3):
    """olympus_qa_studies: synthetic correctness bench."""
    return _finite_blob(olympus_qa_studies.bench_olympus_qa_studies(seed))


def bench_phoenix_qa_studies_family(seed: int = _SEED + 4):
    """phoenix_qa_studies: synthetic correctness bench."""
    return _finite_blob(phoenix_qa_studies.bench_phoenix_qa_studies(seed))


def bench_titan_qa_studies_family(seed: int = _SEED + 5):
    """titan_qa_studies: synthetic correctness bench."""
    return _finite_blob(titan_qa_studies.bench_titan_qa_studies(seed))
