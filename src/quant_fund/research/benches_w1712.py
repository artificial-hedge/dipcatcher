"""Wave-1712 bench adapters: armenian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    astghik_qa_studies,
    hayk_qa_studies,
    nahapet_qa_studies,
    nane_qa_studies,
    tir_qa_studies,
    vahagn_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17120


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_astghik_qa_studies_family(seed: int = _SEED + 0):
    """astghik_qa_studies: synthetic correctness bench."""
    return _finite_blob(astghik_qa_studies.bench_astghik_qa_studies(seed))


def bench_hayk_qa_studies_family(seed: int = _SEED + 1):
    """hayk_qa_studies: synthetic correctness bench."""
    return _finite_blob(hayk_qa_studies.bench_hayk_qa_studies(seed))


def bench_nahapet_qa_studies_family(seed: int = _SEED + 2):
    """nahapet_qa_studies: synthetic correctness bench."""
    return _finite_blob(nahapet_qa_studies.bench_nahapet_qa_studies(seed))


def bench_nane_qa_studies_family(seed: int = _SEED + 3):
    """nane_qa_studies: synthetic correctness bench."""
    return _finite_blob(nane_qa_studies.bench_nane_qa_studies(seed))


def bench_tir_qa_studies_family(seed: int = _SEED + 4):
    """tir_qa_studies: synthetic correctness bench."""
    return _finite_blob(tir_qa_studies.bench_tir_qa_studies(seed))


def bench_vahagn_qa_studies_family(seed: int = _SEED + 5):
    """vahagn_qa_studies: synthetic correctness bench."""
    return _finite_blob(vahagn_qa_studies.bench_vahagn_qa_studies(seed))
