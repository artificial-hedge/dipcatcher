"""Wave-1597 bench adapters: caprine canon (SYNTHETIC only)."""

from quant_fund.models import (
    bharal_qa_studies,
    chamois_qa_studies,
    goral_qa_studies,
    ibex_qa_studies,
    serow_qa_studies,
    tahr_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15970


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


def bench_bharal_qa_studies_family(seed: int = _SEED + 0):
    """bharal_qa_studies: synthetic correctness bench."""
    return _finite_blob(bharal_qa_studies.bench_bharal_qa_studies(seed))


def bench_chamois_qa_studies_family(seed: int = _SEED + 1):
    """chamois_qa_studies: synthetic correctness bench."""
    return _finite_blob(chamois_qa_studies.bench_chamois_qa_studies(seed))


def bench_goral_qa_studies_family(seed: int = _SEED + 2):
    """goral_qa_studies: synthetic correctness bench."""
    return _finite_blob(goral_qa_studies.bench_goral_qa_studies(seed))


def bench_ibex_qa_studies_family(seed: int = _SEED + 3):
    """ibex_qa_studies: synthetic correctness bench."""
    return _finite_blob(ibex_qa_studies.bench_ibex_qa_studies(seed))


def bench_serow_qa_studies_family(seed: int = _SEED + 4):
    """serow_qa_studies: synthetic correctness bench."""
    return _finite_blob(serow_qa_studies.bench_serow_qa_studies(seed))


def bench_tahr_qa_studies_family(seed: int = _SEED + 5):
    """tahr_qa_studies: synthetic correctness bench."""
    return _finite_blob(tahr_qa_studies.bench_tahr_qa_studies(seed))
