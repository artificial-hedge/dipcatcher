"""Wave-1743 bench adapters: zulu-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    impundulu_qa_studies,
    inkanyamba_qa_studies,
    mamlambo_qa_studies,
    tikoloshe_qa_studies,
    unkulunkulu_qa_studies,
    usilosimapundu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17430


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_impundulu_qa_studies_family(seed: int = _SEED + 0):
    """impundulu_qa_studies: synthetic correctness bench."""
    return _finite_blob(impundulu_qa_studies.bench_impundulu_qa_studies(seed))


def bench_inkanyamba_qa_studies_family(seed: int = _SEED + 1):
    """inkanyamba_qa_studies: synthetic correctness bench."""
    return _finite_blob(inkanyamba_qa_studies.bench_inkanyamba_qa_studies(seed))


def bench_mamlambo_qa_studies_family(seed: int = _SEED + 2):
    """mamlambo_qa_studies: synthetic correctness bench."""
    return _finite_blob(mamlambo_qa_studies.bench_mamlambo_qa_studies(seed))


def bench_tikoloshe_qa_studies_family(seed: int = _SEED + 3):
    """tikoloshe_qa_studies: synthetic correctness bench."""
    return _finite_blob(tikoloshe_qa_studies.bench_tikoloshe_qa_studies(seed))


def bench_unkulunkulu_qa_studies_family(seed: int = _SEED + 4):
    """unkulunkulu_qa_studies: synthetic correctness bench."""
    return _finite_blob(unkulunkulu_qa_studies.bench_unkulunkulu_qa_studies(seed))


def bench_usilosimapundu_qa_studies_family(seed: int = _SEED + 5):
    """usilosimapundu_qa_studies: synthetic correctness bench."""
    return _finite_blob(usilosimapundu_qa_studies.bench_usilosimapundu_qa_studies(seed))
