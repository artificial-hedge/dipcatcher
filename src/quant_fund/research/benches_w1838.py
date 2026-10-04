"""Wave-1838 bench adapters: palmyrene-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    aglibol2_qa_studies,
    astarte2_qa_studies,
    baalshamin2_qa_studies,
    bel2_qa_studies,
    malakbel2_qa_studies,
    yarhibol2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18380


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aglibol2_qa_studies_family(seed: int = _SEED + 0):
    """aglibol2_qa_studies: synthetic correctness bench."""
    return _finite_blob(aglibol2_qa_studies.bench_aglibol2_qa_studies(seed))


def bench_astarte2_qa_studies_family(seed: int = _SEED + 1):
    """astarte2_qa_studies: synthetic correctness bench."""
    return _finite_blob(astarte2_qa_studies.bench_astarte2_qa_studies(seed))


def bench_baalshamin2_qa_studies_family(seed: int = _SEED + 2):
    """baalshamin2_qa_studies: synthetic correctness bench."""
    return _finite_blob(baalshamin2_qa_studies.bench_baalshamin2_qa_studies(seed))


def bench_bel2_qa_studies_family(seed: int = _SEED + 3):
    """bel2_qa_studies: synthetic correctness bench."""
    return _finite_blob(bel2_qa_studies.bench_bel2_qa_studies(seed))


def bench_malakbel2_qa_studies_family(seed: int = _SEED + 4):
    """malakbel2_qa_studies: synthetic correctness bench."""
    return _finite_blob(malakbel2_qa_studies.bench_malakbel2_qa_studies(seed))


def bench_yarhibol2_qa_studies_family(seed: int = _SEED + 5):
    """yarhibol2_qa_studies: synthetic correctness bench."""
    return _finite_blob(yarhibol2_qa_studies.bench_yarhibol2_qa_studies(seed))
