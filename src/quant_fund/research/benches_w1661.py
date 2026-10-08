"""Wave-1661 bench adapters: norse-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    draugr_qa_studies,
    fenrir_qa_studies,
    gullinbursti_qa_studies,
    hraesvelgr_qa_studies,
    huginn_qa_studies,
    muninn_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16610


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


def bench_draugr_qa_studies_family(seed: int = _SEED + 0):
    """draugr_qa_studies: synthetic correctness bench."""
    return _finite_blob(draugr_qa_studies.bench_draugr_qa_studies(seed))


def bench_fenrir_qa_studies_family(seed: int = _SEED + 1):
    """fenrir_qa_studies: synthetic correctness bench."""
    return _finite_blob(fenrir_qa_studies.bench_fenrir_qa_studies(seed))


def bench_gullinbursti_qa_studies_family(seed: int = _SEED + 2):
    """gullinbursti_qa_studies: synthetic correctness bench."""
    return _finite_blob(gullinbursti_qa_studies.bench_gullinbursti_qa_studies(seed))


def bench_hraesvelgr_qa_studies_family(seed: int = _SEED + 3):
    """hraesvelgr_qa_studies: synthetic correctness bench."""
    return _finite_blob(hraesvelgr_qa_studies.bench_hraesvelgr_qa_studies(seed))


def bench_huginn_qa_studies_family(seed: int = _SEED + 4):
    """huginn_qa_studies: synthetic correctness bench."""
    return _finite_blob(huginn_qa_studies.bench_huginn_qa_studies(seed))


def bench_muninn_qa_studies_family(seed: int = _SEED + 5):
    """muninn_qa_studies: synthetic correctness bench."""
    return _finite_blob(muninn_qa_studies.bench_muninn_qa_studies(seed))
