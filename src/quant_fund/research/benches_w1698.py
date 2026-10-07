"""Wave-1698 bench adapters: polynesian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    maui_qa_studies,
    menahune_qa_studies,
    pele_qa_studies,
    rangi_qa_studies,
    tane_qa_studies,
    tangaroa_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16980


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


def bench_maui_qa_studies_family(seed: int = _SEED + 0):
    """maui_qa_studies: synthetic correctness bench."""
    return _finite_blob(maui_qa_studies.bench_maui_qa_studies(seed))


def bench_menahune_qa_studies_family(seed: int = _SEED + 1):
    """menahune_qa_studies: synthetic correctness bench."""
    return _finite_blob(menahune_qa_studies.bench_menahune_qa_studies(seed))


def bench_pele_qa_studies_family(seed: int = _SEED + 2):
    """pele_qa_studies: synthetic correctness bench."""
    return _finite_blob(pele_qa_studies.bench_pele_qa_studies(seed))


def bench_rangi_qa_studies_family(seed: int = _SEED + 3):
    """rangi_qa_studies: synthetic correctness bench."""
    return _finite_blob(rangi_qa_studies.bench_rangi_qa_studies(seed))


def bench_tane_qa_studies_family(seed: int = _SEED + 4):
    """tane_qa_studies: synthetic correctness bench."""
    return _finite_blob(tane_qa_studies.bench_tane_qa_studies(seed))


def bench_tangaroa_qa_studies_family(seed: int = _SEED + 5):
    """tangaroa_qa_studies: synthetic correctness bench."""
    return _finite_blob(tangaroa_qa_studies.bench_tangaroa_qa_studies(seed))
