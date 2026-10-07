"""Wave-1469 bench adapters: coastal canon (SYNTHETIC only)."""

from quant_fund.models import (
    atoll_qa_studies,
    bluff_qa_studies,
    cove_qa_studies,
    headland_qa_studies,
    inlet_qa_studies,
    islet_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14690


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


def bench_atoll_qa_studies_family(seed: int = _SEED + 0):
    """atoll_qa_studies: synthetic correctness bench."""
    return _finite_blob(atoll_qa_studies.bench_atoll_qa_studies(seed))


def bench_bluff_qa_studies_family(seed: int = _SEED + 1):
    """bluff_qa_studies: synthetic correctness bench."""
    return _finite_blob(bluff_qa_studies.bench_bluff_qa_studies(seed))


def bench_cove_qa_studies_family(seed: int = _SEED + 2):
    """cove_qa_studies: synthetic correctness bench."""
    return _finite_blob(cove_qa_studies.bench_cove_qa_studies(seed))


def bench_headland_qa_studies_family(seed: int = _SEED + 3):
    """headland_qa_studies: synthetic correctness bench."""
    return _finite_blob(headland_qa_studies.bench_headland_qa_studies(seed))


def bench_inlet_qa_studies_family(seed: int = _SEED + 4):
    """inlet_qa_studies: synthetic correctness bench."""
    return _finite_blob(inlet_qa_studies.bench_inlet_qa_studies(seed))


def bench_islet_qa_studies_family(seed: int = _SEED + 5):
    """islet_qa_studies: synthetic correctness bench."""
    return _finite_blob(islet_qa_studies.bench_islet_qa_studies(seed))
