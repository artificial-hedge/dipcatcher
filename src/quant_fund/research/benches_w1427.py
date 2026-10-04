"""Wave-1427 bench adapters: terrain-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    canyon_qa_studies,
    coast_qa_studies,
    desert_qa_studies,
    field_qa_studies,
    forest_qa_studies,
    glacier_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14270


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_canyon_qa_studies_family(seed: int = _SEED + 0):
    """canyon_qa_studies: synthetic correctness bench."""
    return _finite_blob(canyon_qa_studies.bench_canyon_qa_studies(seed))


def bench_coast_qa_studies_family(seed: int = _SEED + 1):
    """coast_qa_studies: synthetic correctness bench."""
    return _finite_blob(coast_qa_studies.bench_coast_qa_studies(seed))


def bench_desert_qa_studies_family(seed: int = _SEED + 2):
    """desert_qa_studies: synthetic correctness bench."""
    return _finite_blob(desert_qa_studies.bench_desert_qa_studies(seed))


def bench_field_qa_studies_family(seed: int = _SEED + 3):
    """field_qa_studies: synthetic correctness bench."""
    return _finite_blob(field_qa_studies.bench_field_qa_studies(seed))


def bench_forest_qa_studies_family(seed: int = _SEED + 4):
    """forest_qa_studies: synthetic correctness bench."""
    return _finite_blob(forest_qa_studies.bench_forest_qa_studies(seed))


def bench_glacier_qa_studies_family(seed: int = _SEED + 5):
    """glacier_qa_studies: synthetic correctness bench."""
    return _finite_blob(glacier_qa_studies.bench_glacier_qa_studies(seed))
