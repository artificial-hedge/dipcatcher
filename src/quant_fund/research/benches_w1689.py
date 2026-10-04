"""Wave-1689 bench adapters: hindu-myth-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    apsara_qa_studies,
    bhairava_qa_studies,
    bhuta_qa_studies,
    pretas_qa_studies,
    vetal_qa_studies,
    yaksha_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16890


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_apsara_qa_studies_family(seed: int = _SEED + 0):
    """apsara_qa_studies: synthetic correctness bench."""
    return _finite_blob(apsara_qa_studies.bench_apsara_qa_studies(seed))


def bench_bhairava_qa_studies_family(seed: int = _SEED + 1):
    """bhairava_qa_studies: synthetic correctness bench."""
    return _finite_blob(bhairava_qa_studies.bench_bhairava_qa_studies(seed))


def bench_bhuta_qa_studies_family(seed: int = _SEED + 2):
    """bhuta_qa_studies: synthetic correctness bench."""
    return _finite_blob(bhuta_qa_studies.bench_bhuta_qa_studies(seed))


def bench_pretas_qa_studies_family(seed: int = _SEED + 3):
    """pretas_qa_studies: synthetic correctness bench."""
    return _finite_blob(pretas_qa_studies.bench_pretas_qa_studies(seed))


def bench_vetal_qa_studies_family(seed: int = _SEED + 4):
    """vetal_qa_studies: synthetic correctness bench."""
    return _finite_blob(vetal_qa_studies.bench_vetal_qa_studies(seed))


def bench_yaksha_qa_studies_family(seed: int = _SEED + 5):
    """yaksha_qa_studies: synthetic correctness bench."""
    return _finite_blob(yaksha_qa_studies.bench_yaksha_qa_studies(seed))
