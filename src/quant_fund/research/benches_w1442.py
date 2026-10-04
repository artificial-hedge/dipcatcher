"""Wave-1442 bench adapters: insect canon (SYNTHETIC only)."""

from quant_fund.models import (
    ant_qa_studies,
    bee_qa_studies,
    beetle_qa_studies,
    butterfly_qa_studies,
    cricket_qa_studies,
    moth_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14420


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ant_qa_studies_family(seed: int = _SEED + 0):
    """ant_qa_studies: synthetic correctness bench."""
    return _finite_blob(ant_qa_studies.bench_ant_qa_studies(seed))


def bench_bee_qa_studies_family(seed: int = _SEED + 1):
    """bee_qa_studies: synthetic correctness bench."""
    return _finite_blob(bee_qa_studies.bench_bee_qa_studies(seed))


def bench_beetle_qa_studies_family(seed: int = _SEED + 2):
    """beetle_qa_studies: synthetic correctness bench."""
    return _finite_blob(beetle_qa_studies.bench_beetle_qa_studies(seed))


def bench_butterfly_qa_studies_family(seed: int = _SEED + 3):
    """butterfly_qa_studies: synthetic correctness bench."""
    return _finite_blob(butterfly_qa_studies.bench_butterfly_qa_studies(seed))


def bench_cricket_qa_studies_family(seed: int = _SEED + 4):
    """cricket_qa_studies: synthetic correctness bench."""
    return _finite_blob(cricket_qa_studies.bench_cricket_qa_studies(seed))


def bench_moth_qa_studies_family(seed: int = _SEED + 5):
    """moth_qa_studies: synthetic correctness bench."""
    return _finite_blob(moth_qa_studies.bench_moth_qa_studies(seed))
