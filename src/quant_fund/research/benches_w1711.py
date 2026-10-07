"""Wave-1711 bench adapters: phoenician-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    baalat_qa_studies,
    dagon_qa_studies,
    eshmun_qa_studies,
    melqart_qa_studies,
    resheph_qa_studies,
    tanit_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17110


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


def bench_baalat_qa_studies_family(seed: int = _SEED + 0):
    """baalat_qa_studies: synthetic correctness bench."""
    return _finite_blob(baalat_qa_studies.bench_baalat_qa_studies(seed))


def bench_dagon_qa_studies_family(seed: int = _SEED + 1):
    """dagon_qa_studies: synthetic correctness bench."""
    return _finite_blob(dagon_qa_studies.bench_dagon_qa_studies(seed))


def bench_eshmun_qa_studies_family(seed: int = _SEED + 2):
    """eshmun_qa_studies: synthetic correctness bench."""
    return _finite_blob(eshmun_qa_studies.bench_eshmun_qa_studies(seed))


def bench_melqart_qa_studies_family(seed: int = _SEED + 3):
    """melqart_qa_studies: synthetic correctness bench."""
    return _finite_blob(melqart_qa_studies.bench_melqart_qa_studies(seed))


def bench_resheph_qa_studies_family(seed: int = _SEED + 4):
    """resheph_qa_studies: synthetic correctness bench."""
    return _finite_blob(resheph_qa_studies.bench_resheph_qa_studies(seed))


def bench_tanit_qa_studies_family(seed: int = _SEED + 5):
    """tanit_qa_studies: synthetic correctness bench."""
    return _finite_blob(tanit_qa_studies.bench_tanit_qa_studies(seed))
