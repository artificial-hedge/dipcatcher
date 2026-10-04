"""Wave-1438 bench adapters: marine canon (SYNTHETIC only)."""

from quant_fund.models import (
    coral_qa_studies,
    dolphin_qa_studies,
    reef_qa_studies,
    shark_qa_studies,
    turtle_qa_studies,
    whale_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14380


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_coral_qa_studies_family(seed: int = _SEED + 0):
    """coral_qa_studies: synthetic correctness bench."""
    return _finite_blob(coral_qa_studies.bench_coral_qa_studies(seed))


def bench_dolphin_qa_studies_family(seed: int = _SEED + 1):
    """dolphin_qa_studies: synthetic correctness bench."""
    return _finite_blob(dolphin_qa_studies.bench_dolphin_qa_studies(seed))


def bench_reef_qa_studies_family(seed: int = _SEED + 2):
    """reef_qa_studies: synthetic correctness bench."""
    return _finite_blob(reef_qa_studies.bench_reef_qa_studies(seed))


def bench_shark_qa_studies_family(seed: int = _SEED + 3):
    """shark_qa_studies: synthetic correctness bench."""
    return _finite_blob(shark_qa_studies.bench_shark_qa_studies(seed))


def bench_turtle_qa_studies_family(seed: int = _SEED + 4):
    """turtle_qa_studies: synthetic correctness bench."""
    return _finite_blob(turtle_qa_studies.bench_turtle_qa_studies(seed))


def bench_whale_qa_studies_family(seed: int = _SEED + 5):
    """whale_qa_studies: synthetic correctness bench."""
    return _finite_blob(whale_qa_studies.bench_whale_qa_studies(seed))
