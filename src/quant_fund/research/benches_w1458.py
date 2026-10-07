"""Wave-1458 bench adapters: desert-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    arroyo_qa_studies,
    butte_qa_studies,
    camel_qa_studies,
    caravan_qa_studies,
    mirage_qa_studies,
    oasis_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14580


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


def bench_arroyo_qa_studies_family(seed: int = _SEED + 0):
    """arroyo_qa_studies: synthetic correctness bench."""
    return _finite_blob(arroyo_qa_studies.bench_arroyo_qa_studies(seed))


def bench_butte_qa_studies_family(seed: int = _SEED + 1):
    """butte_qa_studies: synthetic correctness bench."""
    return _finite_blob(butte_qa_studies.bench_butte_qa_studies(seed))


def bench_camel_qa_studies_family(seed: int = _SEED + 2):
    """camel_qa_studies: synthetic correctness bench."""
    return _finite_blob(camel_qa_studies.bench_camel_qa_studies(seed))


def bench_caravan_qa_studies_family(seed: int = _SEED + 3):
    """caravan_qa_studies: synthetic correctness bench."""
    return _finite_blob(caravan_qa_studies.bench_caravan_qa_studies(seed))


def bench_mirage_qa_studies_family(seed: int = _SEED + 4):
    """mirage_qa_studies: synthetic correctness bench."""
    return _finite_blob(mirage_qa_studies.bench_mirage_qa_studies(seed))


def bench_oasis_qa_studies_family(seed: int = _SEED + 5):
    """oasis_qa_studies: synthetic correctness bench."""
    return _finite_blob(oasis_qa_studies.bench_oasis_qa_studies(seed))
