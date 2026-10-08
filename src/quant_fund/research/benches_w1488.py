"""Wave-1488 bench adapters: wildcat-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bobcat_qa_studies,
    dingo_qa_studies,
    kodkod_qa_studies,
    oncilla_qa_studies,
    panther_qa_studies,
    tiger_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14880


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


def bench_bobcat_qa_studies_family(seed: int = _SEED + 0):
    """bobcat_qa_studies: synthetic correctness bench."""
    return _finite_blob(bobcat_qa_studies.bench_bobcat_qa_studies(seed))


def bench_dingo_qa_studies_family(seed: int = _SEED + 1):
    """dingo_qa_studies: synthetic correctness bench."""
    return _finite_blob(dingo_qa_studies.bench_dingo_qa_studies(seed))


def bench_kodkod_qa_studies_family(seed: int = _SEED + 2):
    """kodkod_qa_studies: synthetic correctness bench."""
    return _finite_blob(kodkod_qa_studies.bench_kodkod_qa_studies(seed))


def bench_oncilla_qa_studies_family(seed: int = _SEED + 3):
    """oncilla_qa_studies: synthetic correctness bench."""
    return _finite_blob(oncilla_qa_studies.bench_oncilla_qa_studies(seed))


def bench_panther_qa_studies_family(seed: int = _SEED + 4):
    """panther_qa_studies: synthetic correctness bench."""
    return _finite_blob(panther_qa_studies.bench_panther_qa_studies(seed))


def bench_tiger_qa_studies_family(seed: int = _SEED + 5):
    """tiger_qa_studies: synthetic correctness bench."""
    return _finite_blob(tiger_qa_studies.bench_tiger_qa_studies(seed))
