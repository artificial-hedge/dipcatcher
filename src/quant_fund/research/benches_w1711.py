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
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
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
