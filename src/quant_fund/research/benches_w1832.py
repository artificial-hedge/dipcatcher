"""Wave-1832 bench adapters: phoenician-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    baalat2_qa_studies,
    eshmun2_qa_studies,
    melqart2_qa_studies,
    reshef2_qa_studies,
    tanit2_qa_studies,
    yam2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18320


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_baalat2_qa_studies_family(seed: int = _SEED + 0):
    """baalat2_qa_studies: synthetic correctness bench."""
    return _finite_blob(baalat2_qa_studies.bench_baalat2_qa_studies(seed))


def bench_eshmun2_qa_studies_family(seed: int = _SEED + 1):
    """eshmun2_qa_studies: synthetic correctness bench."""
    return _finite_blob(eshmun2_qa_studies.bench_eshmun2_qa_studies(seed))


def bench_melqart2_qa_studies_family(seed: int = _SEED + 2):
    """melqart2_qa_studies: synthetic correctness bench."""
    return _finite_blob(melqart2_qa_studies.bench_melqart2_qa_studies(seed))


def bench_reshef2_qa_studies_family(seed: int = _SEED + 3):
    """reshef2_qa_studies: synthetic correctness bench."""
    return _finite_blob(reshef2_qa_studies.bench_reshef2_qa_studies(seed))


def bench_tanit2_qa_studies_family(seed: int = _SEED + 4):
    """tanit2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tanit2_qa_studies.bench_tanit2_qa_studies(seed))


def bench_yam2_qa_studies_family(seed: int = _SEED + 5):
    """yam2_qa_studies: synthetic correctness bench."""
    return _finite_blob(yam2_qa_studies.bench_yam2_qa_studies(seed))
