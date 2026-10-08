"""Wave-1590 bench adapters: small-cat canon (SYNTHETIC only)."""

from quant_fund.models import (
    black_footed_qa_studies,
    fishing_cat_qa_studies,
    jungle_cat_qa_studies,
    pallas_qa_studies,
    rusty_spotted_qa_studies,
    sand_cat_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15900


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


def bench_black_footed_qa_studies_family(seed: int = _SEED + 0):
    """black_footed_qa_studies: synthetic correctness bench."""
    return _finite_blob(black_footed_qa_studies.bench_black_footed_qa_studies(seed))


def bench_fishing_cat_qa_studies_family(seed: int = _SEED + 1):
    """fishing_cat_qa_studies: synthetic correctness bench."""
    return _finite_blob(fishing_cat_qa_studies.bench_fishing_cat_qa_studies(seed))


def bench_jungle_cat_qa_studies_family(seed: int = _SEED + 2):
    """jungle_cat_qa_studies: synthetic correctness bench."""
    return _finite_blob(jungle_cat_qa_studies.bench_jungle_cat_qa_studies(seed))


def bench_pallas_qa_studies_family(seed: int = _SEED + 3):
    """pallas_qa_studies: synthetic correctness bench."""
    return _finite_blob(pallas_qa_studies.bench_pallas_qa_studies(seed))


def bench_rusty_spotted_qa_studies_family(seed: int = _SEED + 4):
    """rusty_spotted_qa_studies: synthetic correctness bench."""
    return _finite_blob(rusty_spotted_qa_studies.bench_rusty_spotted_qa_studies(seed))


def bench_sand_cat_qa_studies_family(seed: int = _SEED + 5):
    """sand_cat_qa_studies: synthetic correctness bench."""
    return _finite_blob(sand_cat_qa_studies.bench_sand_cat_qa_studies(seed))
