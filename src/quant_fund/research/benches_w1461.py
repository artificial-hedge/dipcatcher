"""Wave-1461 bench adapters: jungle canon (SYNTHETIC only)."""

from quant_fund.models import (
    gorilla_qa_studies,
    jaguar_qa_studies,
    macaw_qa_studies,
    orangutan_qa_studies,
    sloth_qa_studies,
    toucan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14610


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


def bench_gorilla_qa_studies_family(seed: int = _SEED + 0):
    """gorilla_qa_studies: synthetic correctness bench."""
    return _finite_blob(gorilla_qa_studies.bench_gorilla_qa_studies(seed))


def bench_jaguar_qa_studies_family(seed: int = _SEED + 1):
    """jaguar_qa_studies: synthetic correctness bench."""
    return _finite_blob(jaguar_qa_studies.bench_jaguar_qa_studies(seed))


def bench_macaw_qa_studies_family(seed: int = _SEED + 2):
    """macaw_qa_studies: synthetic correctness bench."""
    return _finite_blob(macaw_qa_studies.bench_macaw_qa_studies(seed))


def bench_orangutan_qa_studies_family(seed: int = _SEED + 3):
    """orangutan_qa_studies: synthetic correctness bench."""
    return _finite_blob(orangutan_qa_studies.bench_orangutan_qa_studies(seed))


def bench_sloth_qa_studies_family(seed: int = _SEED + 4):
    """sloth_qa_studies: synthetic correctness bench."""
    return _finite_blob(sloth_qa_studies.bench_sloth_qa_studies(seed))


def bench_toucan_qa_studies_family(seed: int = _SEED + 5):
    """toucan_qa_studies: synthetic correctness bench."""
    return _finite_blob(toucan_qa_studies.bench_toucan_qa_studies(seed))
