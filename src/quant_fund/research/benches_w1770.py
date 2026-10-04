"""Wave-1770 bench adapters: assyrian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    enkidu_qa_studies,
    etana_qa_studies,
    gilgamesh_qa_studies,
    kingu_qa_studies,
    nabu_qa_studies,
    tiamat_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17700


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_enkidu_qa_studies_family(seed: int = _SEED + 0):
    """enkidu_qa_studies: synthetic correctness bench."""
    return _finite_blob(enkidu_qa_studies.bench_enkidu_qa_studies(seed))


def bench_etana_qa_studies_family(seed: int = _SEED + 1):
    """etana_qa_studies: synthetic correctness bench."""
    return _finite_blob(etana_qa_studies.bench_etana_qa_studies(seed))


def bench_gilgamesh_qa_studies_family(seed: int = _SEED + 2):
    """gilgamesh_qa_studies: synthetic correctness bench."""
    return _finite_blob(gilgamesh_qa_studies.bench_gilgamesh_qa_studies(seed))


def bench_kingu_qa_studies_family(seed: int = _SEED + 3):
    """kingu_qa_studies: synthetic correctness bench."""
    return _finite_blob(kingu_qa_studies.bench_kingu_qa_studies(seed))


def bench_nabu_qa_studies_family(seed: int = _SEED + 4):
    """nabu_qa_studies: synthetic correctness bench."""
    return _finite_blob(nabu_qa_studies.bench_nabu_qa_studies(seed))


def bench_tiamat_qa_studies_family(seed: int = _SEED + 5):
    """tiamat_qa_studies: synthetic correctness bench."""
    return _finite_blob(tiamat_qa_studies.bench_tiamat_qa_studies(seed))
