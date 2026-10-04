"""Wave-1587 bench adapters: forest-deer canon (SYNTHETIC only)."""

from quant_fund.models import (
    barasingha_qa_studies,
    brocket_qa_studies,
    huemul_qa_studies,
    mule_deer_qa_studies,
    sambar_qa_studies,
    taruca_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15870


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_barasingha_qa_studies_family(seed: int = _SEED + 0):
    """barasingha_qa_studies: synthetic correctness bench."""
    return _finite_blob(barasingha_qa_studies.bench_barasingha_qa_studies(seed))


def bench_brocket_qa_studies_family(seed: int = _SEED + 1):
    """brocket_qa_studies: synthetic correctness bench."""
    return _finite_blob(brocket_qa_studies.bench_brocket_qa_studies(seed))


def bench_huemul_qa_studies_family(seed: int = _SEED + 2):
    """huemul_qa_studies: synthetic correctness bench."""
    return _finite_blob(huemul_qa_studies.bench_huemul_qa_studies(seed))


def bench_mule_deer_qa_studies_family(seed: int = _SEED + 3):
    """mule_deer_qa_studies: synthetic correctness bench."""
    return _finite_blob(mule_deer_qa_studies.bench_mule_deer_qa_studies(seed))


def bench_sambar_qa_studies_family(seed: int = _SEED + 4):
    """sambar_qa_studies: synthetic correctness bench."""
    return _finite_blob(sambar_qa_studies.bench_sambar_qa_studies(seed))


def bench_taruca_qa_studies_family(seed: int = _SEED + 5):
    """taruca_qa_studies: synthetic correctness bench."""
    return _finite_blob(taruca_qa_studies.bench_taruca_qa_studies(seed))
