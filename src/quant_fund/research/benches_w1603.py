"""Wave-1603 bench adapters: deer-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    axis_qa_studies,
    marsh_deer_qa_studies,
    musk_deer_qa_studies,
    pampas_deer_qa_studies,
    tufted_qa_studies,
    water_deer_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16030


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_axis_qa_studies_family(seed: int = _SEED + 0):
    """axis_qa_studies: synthetic correctness bench."""
    return _finite_blob(axis_qa_studies.bench_axis_qa_studies(seed))


def bench_marsh_deer_qa_studies_family(seed: int = _SEED + 1):
    """marsh_deer_qa_studies: synthetic correctness bench."""
    return _finite_blob(marsh_deer_qa_studies.bench_marsh_deer_qa_studies(seed))


def bench_musk_deer_qa_studies_family(seed: int = _SEED + 2):
    """musk_deer_qa_studies: synthetic correctness bench."""
    return _finite_blob(musk_deer_qa_studies.bench_musk_deer_qa_studies(seed))


def bench_pampas_deer_qa_studies_family(seed: int = _SEED + 3):
    """pampas_deer_qa_studies: synthetic correctness bench."""
    return _finite_blob(pampas_deer_qa_studies.bench_pampas_deer_qa_studies(seed))


def bench_tufted_qa_studies_family(seed: int = _SEED + 4):
    """tufted_qa_studies: synthetic correctness bench."""
    return _finite_blob(tufted_qa_studies.bench_tufted_qa_studies(seed))


def bench_water_deer_qa_studies_family(seed: int = _SEED + 5):
    """water_deer_qa_studies: synthetic correctness bench."""
    return _finite_blob(water_deer_qa_studies.bench_water_deer_qa_studies(seed))
