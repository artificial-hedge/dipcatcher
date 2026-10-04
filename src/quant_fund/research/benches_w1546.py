"""Wave-1546 bench adapters: rail-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    corncrake_qa_studies,
    flufftail_qa_studies,
    sora_qa_studies,
    sungrebe_qa_studies,
    swamphen_qa_studies,
    takhe_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15460


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_corncrake_qa_studies_family(seed: int = _SEED + 0):
    """corncrake_qa_studies: synthetic correctness bench."""
    return _finite_blob(corncrake_qa_studies.bench_corncrake_qa_studies(seed))


def bench_flufftail_qa_studies_family(seed: int = _SEED + 1):
    """flufftail_qa_studies: synthetic correctness bench."""
    return _finite_blob(flufftail_qa_studies.bench_flufftail_qa_studies(seed))


def bench_sora_qa_studies_family(seed: int = _SEED + 2):
    """sora_qa_studies: synthetic correctness bench."""
    return _finite_blob(sora_qa_studies.bench_sora_qa_studies(seed))


def bench_sungrebe_qa_studies_family(seed: int = _SEED + 3):
    """sungrebe_qa_studies: synthetic correctness bench."""
    return _finite_blob(sungrebe_qa_studies.bench_sungrebe_qa_studies(seed))


def bench_swamphen_qa_studies_family(seed: int = _SEED + 4):
    """swamphen_qa_studies: synthetic correctness bench."""
    return _finite_blob(swamphen_qa_studies.bench_swamphen_qa_studies(seed))


def bench_takhe_qa_studies_family(seed: int = _SEED + 5):
    """takhe_qa_studies: synthetic correctness bench."""
    return _finite_blob(takhe_qa_studies.bench_takhe_qa_studies(seed))
