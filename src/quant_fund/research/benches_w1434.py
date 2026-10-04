"""Wave-1434 bench adapters: material canon (SYNTHETIC only)."""

from quant_fund.models import (
    alloy_qa_studies,
    ceramic_qa_studies,
    glass_qa_studies,
    iron_qa_studies,
    steel_qa_studies,
    wood_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14340


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alloy_qa_studies_family(seed: int = _SEED + 0):
    """alloy_qa_studies: synthetic correctness bench."""
    return _finite_blob(alloy_qa_studies.bench_alloy_qa_studies(seed))


def bench_ceramic_qa_studies_family(seed: int = _SEED + 1):
    """ceramic_qa_studies: synthetic correctness bench."""
    return _finite_blob(ceramic_qa_studies.bench_ceramic_qa_studies(seed))


def bench_glass_qa_studies_family(seed: int = _SEED + 2):
    """glass_qa_studies: synthetic correctness bench."""
    return _finite_blob(glass_qa_studies.bench_glass_qa_studies(seed))


def bench_iron_qa_studies_family(seed: int = _SEED + 3):
    """iron_qa_studies: synthetic correctness bench."""
    return _finite_blob(iron_qa_studies.bench_iron_qa_studies(seed))


def bench_steel_qa_studies_family(seed: int = _SEED + 4):
    """steel_qa_studies: synthetic correctness bench."""
    return _finite_blob(steel_qa_studies.bench_steel_qa_studies(seed))


def bench_wood_qa_studies_family(seed: int = _SEED + 5):
    """wood_qa_studies: synthetic correctness bench."""
    return _finite_blob(wood_qa_studies.bench_wood_qa_studies(seed))
