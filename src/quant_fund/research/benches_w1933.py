"""Wave-1933 bench adapters: caribbean-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    bacoo_qa_studies,
    duppy_qa_studies,
    jumbie_qa_studies,
    lagahoo_qa_studies,
    ole_higue_qa_studies,
    soucouyant_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19330


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bacoo_qa_studies_family(seed: int = _SEED + 0):
    """bacoo_qa_studies: synthetic correctness bench."""
    return _finite_blob(bacoo_qa_studies.bench_bacoo_qa_studies(seed))


def bench_duppy_qa_studies_family(seed: int = _SEED + 1):
    """duppy_qa_studies: synthetic correctness bench."""
    return _finite_blob(duppy_qa_studies.bench_duppy_qa_studies(seed))


def bench_jumbie_qa_studies_family(seed: int = _SEED + 2):
    """jumbie_qa_studies: synthetic correctness bench."""
    return _finite_blob(jumbie_qa_studies.bench_jumbie_qa_studies(seed))


def bench_lagahoo_qa_studies_family(seed: int = _SEED + 3):
    """lagahoo_qa_studies: synthetic correctness bench."""
    return _finite_blob(lagahoo_qa_studies.bench_lagahoo_qa_studies(seed))


def bench_ole_higue_qa_studies_family(seed: int = _SEED + 4):
    """ole_higue_qa_studies: synthetic correctness bench."""
    return _finite_blob(ole_higue_qa_studies.bench_ole_higue_qa_studies(seed))


def bench_soucouyant_qa_studies_family(seed: int = _SEED + 5):
    """soucouyant_qa_studies: synthetic correctness bench."""
    return _finite_blob(soucouyant_qa_studies.bench_soucouyant_qa_studies(seed))
