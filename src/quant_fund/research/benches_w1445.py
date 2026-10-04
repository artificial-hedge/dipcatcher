"""Wave-1445 bench adapters: arboreal canon (SYNTHETIC only)."""

from quant_fund.models import (
    birch_qa_studies,
    cedar_qa_studies,
    elm_qa_studies,
    maple_qa_studies,
    oak_qa_studies,
    willow_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14450


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_birch_qa_studies_family(seed: int = _SEED + 0):
    """birch_qa_studies: synthetic correctness bench."""
    return _finite_blob(birch_qa_studies.bench_birch_qa_studies(seed))


def bench_cedar_qa_studies_family(seed: int = _SEED + 1):
    """cedar_qa_studies: synthetic correctness bench."""
    return _finite_blob(cedar_qa_studies.bench_cedar_qa_studies(seed))


def bench_elm_qa_studies_family(seed: int = _SEED + 2):
    """elm_qa_studies: synthetic correctness bench."""
    return _finite_blob(elm_qa_studies.bench_elm_qa_studies(seed))


def bench_maple_qa_studies_family(seed: int = _SEED + 3):
    """maple_qa_studies: synthetic correctness bench."""
    return _finite_blob(maple_qa_studies.bench_maple_qa_studies(seed))


def bench_oak_qa_studies_family(seed: int = _SEED + 4):
    """oak_qa_studies: synthetic correctness bench."""
    return _finite_blob(oak_qa_studies.bench_oak_qa_studies(seed))


def bench_willow_qa_studies_family(seed: int = _SEED + 5):
    """willow_qa_studies: synthetic correctness bench."""
    return _finite_blob(willow_qa_studies.bench_willow_qa_studies(seed))
