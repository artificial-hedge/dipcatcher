"""Wave-1472 bench adapters: herb canon (SYNTHETIC only)."""

from quant_fund.models import (
    basil_qa_studies,
    cardamom_qa_studies,
    chervil_qa_studies,
    cinnamon_qa_studies,
    coriander_qa_studies,
    cumin_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14720


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


def bench_basil_qa_studies_family(seed: int = _SEED + 0):
    """basil_qa_studies: synthetic correctness bench."""
    return _finite_blob(basil_qa_studies.bench_basil_qa_studies(seed))


def bench_cardamom_qa_studies_family(seed: int = _SEED + 1):
    """cardamom_qa_studies: synthetic correctness bench."""
    return _finite_blob(cardamom_qa_studies.bench_cardamom_qa_studies(seed))


def bench_chervil_qa_studies_family(seed: int = _SEED + 2):
    """chervil_qa_studies: synthetic correctness bench."""
    return _finite_blob(chervil_qa_studies.bench_chervil_qa_studies(seed))


def bench_cinnamon_qa_studies_family(seed: int = _SEED + 3):
    """cinnamon_qa_studies: synthetic correctness bench."""
    return _finite_blob(cinnamon_qa_studies.bench_cinnamon_qa_studies(seed))


def bench_coriander_qa_studies_family(seed: int = _SEED + 4):
    """coriander_qa_studies: synthetic correctness bench."""
    return _finite_blob(coriander_qa_studies.bench_coriander_qa_studies(seed))


def bench_cumin_qa_studies_family(seed: int = _SEED + 5):
    """cumin_qa_studies: synthetic correctness bench."""
    return _finite_blob(cumin_qa_studies.bench_cumin_qa_studies(seed))
