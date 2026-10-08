"""Wave-1501 bench adapters: tree-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    acacia_qa_studies,
    alder_qa_studies,
    baobab_qa_studies,
    olive_qa_studies,
    palm_qa_studies,
    sycamore_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15010


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


def bench_acacia_qa_studies_family(seed: int = _SEED + 0):
    """acacia_qa_studies: synthetic correctness bench."""
    return _finite_blob(acacia_qa_studies.bench_acacia_qa_studies(seed))


def bench_alder_qa_studies_family(seed: int = _SEED + 1):
    """alder_qa_studies: synthetic correctness bench."""
    return _finite_blob(alder_qa_studies.bench_alder_qa_studies(seed))


def bench_baobab_qa_studies_family(seed: int = _SEED + 2):
    """baobab_qa_studies: synthetic correctness bench."""
    return _finite_blob(baobab_qa_studies.bench_baobab_qa_studies(seed))


def bench_olive_qa_studies_family(seed: int = _SEED + 3):
    """olive_qa_studies: synthetic correctness bench."""
    return _finite_blob(olive_qa_studies.bench_olive_qa_studies(seed))


def bench_palm_qa_studies_family(seed: int = _SEED + 4):
    """palm_qa_studies: synthetic correctness bench."""
    return _finite_blob(palm_qa_studies.bench_palm_qa_studies(seed))


def bench_sycamore_qa_studies_family(seed: int = _SEED + 5):
    """sycamore_qa_studies: synthetic correctness bench."""
    return _finite_blob(sycamore_qa_studies.bench_sycamore_qa_studies(seed))
