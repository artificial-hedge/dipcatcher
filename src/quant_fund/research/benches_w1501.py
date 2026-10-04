"""Wave-1501 bench adapters: tree-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    acacia_qa_studies,
    alder_qa_studies,
    baobab_qa_studies,
    olive_qa_studies,
    palm_qa_studies,
    sycamore_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15010


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_acacia_qa_studies_family(seed: int = _SEED + 0):
    """acacia_qa_studies: synthetic correctness bench."""
    return _finite_blob(acacia_qa_studies.bench_acacia_qa_studies(seed))


def bench_alder_qa_studies_family(seed: int = _SEED + 1):
    """alder_qa_studies: synthetic correctness bench."""
    return _finite_blob(alder_qa_studies.bench_alder_qa_studies(seed))


def bench_baobab_qa_studies_family(seed: int = _SEED + 2):
    """baobab_qa_studies: synthetic correctness bench."""
    return _finite_blob(baobab_qa_studies.bench_baobab_qa_studies(seed))


def bench_olive_qa_studies_family(seed: int = _SEED + 3):
    """olive_qa_studies: synthetic correctness bench."""
    return _finite_blob(olive_qa_studies.bench_olive_qa_studies(seed))


def bench_palm_qa_studies_family(seed: int = _SEED + 4):
    """palm_qa_studies: synthetic correctness bench."""
    return _finite_blob(palm_qa_studies.bench_palm_qa_studies(seed))


def bench_sycamore_qa_studies_family(seed: int = _SEED + 5):
    """sycamore_qa_studies: synthetic correctness bench."""
    return _finite_blob(sycamore_qa_studies.bench_sycamore_qa_studies(seed))
