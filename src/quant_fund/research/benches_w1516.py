"""Wave-1516 bench adapters: moth canon (SYNTHETIC only)."""

from quant_fund.models import (
    atlas_moth_qa_studies,
    gypsy_moth_qa_studies,
    hawk_moth_qa_studies,
    luna_moth_qa_studies,
    tussock_moth_qa_studies,
    underwing_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15160


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


def bench_atlas_moth_qa_studies_family(seed: int = _SEED + 0):
    """atlas_moth_qa_studies: synthetic correctness bench."""
    return _finite_blob(atlas_moth_qa_studies.bench_atlas_moth_qa_studies(seed))


def bench_gypsy_moth_qa_studies_family(seed: int = _SEED + 1):
    """gypsy_moth_qa_studies: synthetic correctness bench."""
    return _finite_blob(gypsy_moth_qa_studies.bench_gypsy_moth_qa_studies(seed))


def bench_hawk_moth_qa_studies_family(seed: int = _SEED + 2):
    """hawk_moth_qa_studies: synthetic correctness bench."""
    return _finite_blob(hawk_moth_qa_studies.bench_hawk_moth_qa_studies(seed))


def bench_luna_moth_qa_studies_family(seed: int = _SEED + 3):
    """luna_moth_qa_studies: synthetic correctness bench."""
    return _finite_blob(luna_moth_qa_studies.bench_luna_moth_qa_studies(seed))


def bench_tussock_moth_qa_studies_family(seed: int = _SEED + 4):
    """tussock_moth_qa_studies: synthetic correctness bench."""
    return _finite_blob(tussock_moth_qa_studies.bench_tussock_moth_qa_studies(seed))


def bench_underwing_qa_studies_family(seed: int = _SEED + 5):
    """underwing_qa_studies: synthetic correctness bench."""
    return _finite_blob(underwing_qa_studies.bench_underwing_qa_studies(seed))
