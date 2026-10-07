"""Wave-1451 bench adapters: fruit canon (SYNTHETIC only)."""

from quant_fund.models import (
    apple_qa_studies,
    cherry_qa_studies,
    grape_qa_studies,
    lemon_qa_studies,
    mango_qa_studies,
    peach_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14510


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


def bench_apple_qa_studies_family(seed: int = _SEED + 0):
    """apple_qa_studies: synthetic correctness bench."""
    return _finite_blob(apple_qa_studies.bench_apple_qa_studies(seed))


def bench_cherry_qa_studies_family(seed: int = _SEED + 1):
    """cherry_qa_studies: synthetic correctness bench."""
    return _finite_blob(cherry_qa_studies.bench_cherry_qa_studies(seed))


def bench_grape_qa_studies_family(seed: int = _SEED + 2):
    """grape_qa_studies: synthetic correctness bench."""
    return _finite_blob(grape_qa_studies.bench_grape_qa_studies(seed))


def bench_lemon_qa_studies_family(seed: int = _SEED + 3):
    """lemon_qa_studies: synthetic correctness bench."""
    return _finite_blob(lemon_qa_studies.bench_lemon_qa_studies(seed))


def bench_mango_qa_studies_family(seed: int = _SEED + 4):
    """mango_qa_studies: synthetic correctness bench."""
    return _finite_blob(mango_qa_studies.bench_mango_qa_studies(seed))


def bench_peach_qa_studies_family(seed: int = _SEED + 5):
    """peach_qa_studies: synthetic correctness bench."""
    return _finite_blob(peach_qa_studies.bench_peach_qa_studies(seed))
