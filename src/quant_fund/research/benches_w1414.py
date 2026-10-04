"""Wave-1414 bench adapters: folk-commonsense canon (SYNTHETIC only)."""

from quant_fund.models import (
    afford_qa_studies,
    counter_qa_studies,
    custom_qa_studies,
    everyday_qa_studies,
    folk_qa_studies,
    moral_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14140


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_afford_qa_studies_family(seed: int = _SEED + 0):
    """afford_qa_studies: synthetic correctness bench."""
    return _finite_blob(afford_qa_studies.bench_afford_qa_studies(seed))


def bench_counter_qa_studies_family(seed: int = _SEED + 1):
    """counter_qa_studies: synthetic correctness bench."""
    return _finite_blob(counter_qa_studies.bench_counter_qa_studies(seed))


def bench_custom_qa_studies_family(seed: int = _SEED + 2):
    """custom_qa_studies: synthetic correctness bench."""
    return _finite_blob(custom_qa_studies.bench_custom_qa_studies(seed))


def bench_everyday_qa_studies_family(seed: int = _SEED + 3):
    """everyday_qa_studies: synthetic correctness bench."""
    return _finite_blob(everyday_qa_studies.bench_everyday_qa_studies(seed))


def bench_folk_qa_studies_family(seed: int = _SEED + 4):
    """folk_qa_studies: synthetic correctness bench."""
    return _finite_blob(folk_qa_studies.bench_folk_qa_studies(seed))


def bench_moral_qa_studies_family(seed: int = _SEED + 5):
    """moral_qa_studies: synthetic correctness bench."""
    return _finite_blob(moral_qa_studies.bench_moral_qa_studies(seed))
