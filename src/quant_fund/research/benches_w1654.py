"""Wave-1654 bench adapters: bestiary-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    amphisbaena_qa_studies,
    bonnacon_qa_studies,
    cerastes_qa_studies,
    leucrotta_qa_studies,
    parandrus_qa_studies,
    questing_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16540


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


def bench_amphisbaena_qa_studies_family(seed: int = _SEED + 0):
    """amphisbaena_qa_studies: synthetic correctness bench."""
    return _finite_blob(amphisbaena_qa_studies.bench_amphisbaena_qa_studies(seed))


def bench_bonnacon_qa_studies_family(seed: int = _SEED + 1):
    """bonnacon_qa_studies: synthetic correctness bench."""
    return _finite_blob(bonnacon_qa_studies.bench_bonnacon_qa_studies(seed))


def bench_cerastes_qa_studies_family(seed: int = _SEED + 2):
    """cerastes_qa_studies: synthetic correctness bench."""
    return _finite_blob(cerastes_qa_studies.bench_cerastes_qa_studies(seed))


def bench_leucrotta_qa_studies_family(seed: int = _SEED + 3):
    """leucrotta_qa_studies: synthetic correctness bench."""
    return _finite_blob(leucrotta_qa_studies.bench_leucrotta_qa_studies(seed))


def bench_parandrus_qa_studies_family(seed: int = _SEED + 4):
    """parandrus_qa_studies: synthetic correctness bench."""
    return _finite_blob(parandrus_qa_studies.bench_parandrus_qa_studies(seed))


def bench_questing_qa_studies_family(seed: int = _SEED + 5):
    """questing_qa_studies: synthetic correctness bench."""
    return _finite_blob(questing_qa_studies.bench_questing_qa_studies(seed))
