"""Wave-1444 bench adapters: wetland canon (SYNTHETIC only)."""

from quant_fund.models import (
    brook_qa_studies,
    creek_qa_studies,
    delta_qa_studies,
    estuary_qa_studies,
    marsh_qa_studies,
    pond_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14440


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


def bench_brook_qa_studies_family(seed: int = _SEED + 0):
    """brook_qa_studies: synthetic correctness bench."""
    return _finite_blob(brook_qa_studies.bench_brook_qa_studies(seed))


def bench_creek_qa_studies_family(seed: int = _SEED + 1):
    """creek_qa_studies: synthetic correctness bench."""
    return _finite_blob(creek_qa_studies.bench_creek_qa_studies(seed))


def bench_delta_qa_studies_family(seed: int = _SEED + 2):
    """delta_qa_studies: synthetic correctness bench."""
    return _finite_blob(delta_qa_studies.bench_delta_qa_studies(seed))


def bench_estuary_qa_studies_family(seed: int = _SEED + 3):
    """estuary_qa_studies: synthetic correctness bench."""
    return _finite_blob(estuary_qa_studies.bench_estuary_qa_studies(seed))


def bench_marsh_qa_studies_family(seed: int = _SEED + 4):
    """marsh_qa_studies: synthetic correctness bench."""
    return _finite_blob(marsh_qa_studies.bench_marsh_qa_studies(seed))


def bench_pond_qa_studies_family(seed: int = _SEED + 5):
    """pond_qa_studies: synthetic correctness bench."""
    return _finite_blob(pond_qa_studies.bench_pond_qa_studies(seed))
