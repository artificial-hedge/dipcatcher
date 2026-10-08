"""Wave-1432 bench adapters: cuisine canon (SYNTHETIC only)."""

from quant_fund.models import (
    beverage_qa_studies,
    cuisine_qa_studies,
    dessert_qa_studies,
    dish_qa_studies,
    fruit_qa_studies,
    ingredient_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14320


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


def bench_beverage_qa_studies_family(seed: int = _SEED + 0):
    """beverage_qa_studies: synthetic correctness bench."""
    return _finite_blob(beverage_qa_studies.bench_beverage_qa_studies(seed))


def bench_cuisine_qa_studies_family(seed: int = _SEED + 1):
    """cuisine_qa_studies: synthetic correctness bench."""
    return _finite_blob(cuisine_qa_studies.bench_cuisine_qa_studies(seed))


def bench_dessert_qa_studies_family(seed: int = _SEED + 2):
    """dessert_qa_studies: synthetic correctness bench."""
    return _finite_blob(dessert_qa_studies.bench_dessert_qa_studies(seed))


def bench_dish_qa_studies_family(seed: int = _SEED + 3):
    """dish_qa_studies: synthetic correctness bench."""
    return _finite_blob(dish_qa_studies.bench_dish_qa_studies(seed))


def bench_fruit_qa_studies_family(seed: int = _SEED + 4):
    """fruit_qa_studies: synthetic correctness bench."""
    return _finite_blob(fruit_qa_studies.bench_fruit_qa_studies(seed))


def bench_ingredient_qa_studies_family(seed: int = _SEED + 5):
    """ingredient_qa_studies: synthetic correctness bench."""
    return _finite_blob(ingredient_qa_studies.bench_ingredient_qa_studies(seed))
