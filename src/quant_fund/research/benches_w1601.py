"""Wave-1601 bench adapters: bat-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    blossom_bat_qa_studies,
    bulldog_bat_qa_studies,
    free_tailed_qa_studies,
    fruit_bat_qa_studies,
    mouse_eared_qa_studies,
    tent_bat_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16010


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


def bench_blossom_bat_qa_studies_family(seed: int = _SEED + 0):
    """blossom_bat_qa_studies: synthetic correctness bench."""
    return _finite_blob(blossom_bat_qa_studies.bench_blossom_bat_qa_studies(seed))


def bench_bulldog_bat_qa_studies_family(seed: int = _SEED + 1):
    """bulldog_bat_qa_studies: synthetic correctness bench."""
    return _finite_blob(bulldog_bat_qa_studies.bench_bulldog_bat_qa_studies(seed))


def bench_free_tailed_qa_studies_family(seed: int = _SEED + 2):
    """free_tailed_qa_studies: synthetic correctness bench."""
    return _finite_blob(free_tailed_qa_studies.bench_free_tailed_qa_studies(seed))


def bench_fruit_bat_qa_studies_family(seed: int = _SEED + 3):
    """fruit_bat_qa_studies: synthetic correctness bench."""
    return _finite_blob(fruit_bat_qa_studies.bench_fruit_bat_qa_studies(seed))


def bench_mouse_eared_qa_studies_family(seed: int = _SEED + 4):
    """mouse_eared_qa_studies: synthetic correctness bench."""
    return _finite_blob(mouse_eared_qa_studies.bench_mouse_eared_qa_studies(seed))


def bench_tent_bat_qa_studies_family(seed: int = _SEED + 5):
    """tent_bat_qa_studies: synthetic correctness bench."""
    return _finite_blob(tent_bat_qa_studies.bench_tent_bat_qa_studies(seed))
