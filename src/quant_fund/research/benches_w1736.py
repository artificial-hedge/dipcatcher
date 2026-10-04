"""Wave-1736 bench adapters: finnish-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ilmarinen_qa_studies,
    joukahainen_qa_studies,
    lemminkainen_qa_studies,
    marjatta_qa_studies,
    tuoni_qa_studies,
    vainamoinen_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17360


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ilmarinen_qa_studies_family(seed: int = _SEED + 0):
    """ilmarinen_qa_studies: synthetic correctness bench."""
    return _finite_blob(ilmarinen_qa_studies.bench_ilmarinen_qa_studies(seed))


def bench_joukahainen_qa_studies_family(seed: int = _SEED + 1):
    """joukahainen_qa_studies: synthetic correctness bench."""
    return _finite_blob(joukahainen_qa_studies.bench_joukahainen_qa_studies(seed))


def bench_lemminkainen_qa_studies_family(seed: int = _SEED + 2):
    """lemminkainen_qa_studies: synthetic correctness bench."""
    return _finite_blob(lemminkainen_qa_studies.bench_lemminkainen_qa_studies(seed))


def bench_marjatta_qa_studies_family(seed: int = _SEED + 3):
    """marjatta_qa_studies: synthetic correctness bench."""
    return _finite_blob(marjatta_qa_studies.bench_marjatta_qa_studies(seed))


def bench_tuoni_qa_studies_family(seed: int = _SEED + 4):
    """tuoni_qa_studies: synthetic correctness bench."""
    return _finite_blob(tuoni_qa_studies.bench_tuoni_qa_studies(seed))


def bench_vainamoinen_qa_studies_family(seed: int = _SEED + 5):
    """vainamoinen_qa_studies: synthetic correctness bench."""
    return _finite_blob(vainamoinen_qa_studies.bench_vainamoinen_qa_studies(seed))
