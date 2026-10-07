"""Wave-1497 bench adapters: marsupial-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bettong_qa_studies,
    cuscus_qa_studies,
    numbat2_qa_studies,
    pademelon_qa_studies,
    potoroo_qa_studies,
    woylie_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14970


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


def bench_bettong_qa_studies_family(seed: int = _SEED + 0):
    """bettong_qa_studies: synthetic correctness bench."""
    return _finite_blob(bettong_qa_studies.bench_bettong_qa_studies(seed))


def bench_cuscus_qa_studies_family(seed: int = _SEED + 1):
    """cuscus_qa_studies: synthetic correctness bench."""
    return _finite_blob(cuscus_qa_studies.bench_cuscus_qa_studies(seed))


def bench_numbat2_qa_studies_family(seed: int = _SEED + 2):
    """numbat2_qa_studies: synthetic correctness bench."""
    return _finite_blob(numbat2_qa_studies.bench_numbat2_qa_studies(seed))


def bench_pademelon_qa_studies_family(seed: int = _SEED + 3):
    """pademelon_qa_studies: synthetic correctness bench."""
    return _finite_blob(pademelon_qa_studies.bench_pademelon_qa_studies(seed))


def bench_potoroo_qa_studies_family(seed: int = _SEED + 4):
    """potoroo_qa_studies: synthetic correctness bench."""
    return _finite_blob(potoroo_qa_studies.bench_potoroo_qa_studies(seed))


def bench_woylie_qa_studies_family(seed: int = _SEED + 5):
    """woylie_qa_studies: synthetic correctness bench."""
    return _finite_blob(woylie_qa_studies.bench_woylie_qa_studies(seed))
