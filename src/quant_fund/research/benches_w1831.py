"""Wave-1831 bench adapters: canaanite-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    asherah3_qa_studies,
    baal3_qa_studies,
    el3_qa_studies,
    kothar3_qa_studies,
    lotan3_qa_studies,
    mot3_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18310


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_asherah3_qa_studies_family(seed: int = _SEED + 0):
    """asherah3_qa_studies: synthetic correctness bench."""
    return _finite_blob(asherah3_qa_studies.bench_asherah3_qa_studies(seed))


def bench_baal3_qa_studies_family(seed: int = _SEED + 1):
    """baal3_qa_studies: synthetic correctness bench."""
    return _finite_blob(baal3_qa_studies.bench_baal3_qa_studies(seed))


def bench_el3_qa_studies_family(seed: int = _SEED + 2):
    """el3_qa_studies: synthetic correctness bench."""
    return _finite_blob(el3_qa_studies.bench_el3_qa_studies(seed))


def bench_kothar3_qa_studies_family(seed: int = _SEED + 3):
    """kothar3_qa_studies: synthetic correctness bench."""
    return _finite_blob(kothar3_qa_studies.bench_kothar3_qa_studies(seed))


def bench_lotan3_qa_studies_family(seed: int = _SEED + 4):
    """lotan3_qa_studies: synthetic correctness bench."""
    return _finite_blob(lotan3_qa_studies.bench_lotan3_qa_studies(seed))


def bench_mot3_qa_studies_family(seed: int = _SEED + 5):
    """mot3_qa_studies: synthetic correctness bench."""
    return _finite_blob(mot3_qa_studies.bench_mot3_qa_studies(seed))
