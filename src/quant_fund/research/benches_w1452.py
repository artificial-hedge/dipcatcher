"""Wave-1452 bench adapters: vegetable canon (SYNTHETIC only)."""

from quant_fund.models import (
    carrot_qa_studies,
    cucumber_qa_studies,
    garlic_qa_studies,
    onion_qa_studies,
    potato_qa_studies,
    tomato_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14520


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_carrot_qa_studies_family(seed: int = _SEED + 0):
    """carrot_qa_studies: synthetic correctness bench."""
    return _finite_blob(carrot_qa_studies.bench_carrot_qa_studies(seed))


def bench_cucumber_qa_studies_family(seed: int = _SEED + 1):
    """cucumber_qa_studies: synthetic correctness bench."""
    return _finite_blob(cucumber_qa_studies.bench_cucumber_qa_studies(seed))


def bench_garlic_qa_studies_family(seed: int = _SEED + 2):
    """garlic_qa_studies: synthetic correctness bench."""
    return _finite_blob(garlic_qa_studies.bench_garlic_qa_studies(seed))


def bench_onion_qa_studies_family(seed: int = _SEED + 3):
    """onion_qa_studies: synthetic correctness bench."""
    return _finite_blob(onion_qa_studies.bench_onion_qa_studies(seed))


def bench_potato_qa_studies_family(seed: int = _SEED + 4):
    """potato_qa_studies: synthetic correctness bench."""
    return _finite_blob(potato_qa_studies.bench_potato_qa_studies(seed))


def bench_tomato_qa_studies_family(seed: int = _SEED + 5):
    """tomato_qa_studies: synthetic correctness bench."""
    return _finite_blob(tomato_qa_studies.bench_tomato_qa_studies(seed))
