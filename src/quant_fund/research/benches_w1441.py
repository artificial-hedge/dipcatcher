"""Wave-1441 bench adapters: avian canon (SYNTHETIC only)."""

from quant_fund.models import (
    crane_qa_studies,
    eagle_qa_studies,
    falcon_qa_studies,
    owl_qa_studies,
    raven_qa_studies,
    swan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14410


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_crane_qa_studies_family(seed: int = _SEED + 0):
    """crane_qa_studies: synthetic correctness bench."""
    return _finite_blob(crane_qa_studies.bench_crane_qa_studies(seed))


def bench_eagle_qa_studies_family(seed: int = _SEED + 1):
    """eagle_qa_studies: synthetic correctness bench."""
    return _finite_blob(eagle_qa_studies.bench_eagle_qa_studies(seed))


def bench_falcon_qa_studies_family(seed: int = _SEED + 2):
    """falcon_qa_studies: synthetic correctness bench."""
    return _finite_blob(falcon_qa_studies.bench_falcon_qa_studies(seed))


def bench_owl_qa_studies_family(seed: int = _SEED + 3):
    """owl_qa_studies: synthetic correctness bench."""
    return _finite_blob(owl_qa_studies.bench_owl_qa_studies(seed))


def bench_raven_qa_studies_family(seed: int = _SEED + 4):
    """raven_qa_studies: synthetic correctness bench."""
    return _finite_blob(raven_qa_studies.bench_raven_qa_studies(seed))


def bench_swan_qa_studies_family(seed: int = _SEED + 5):
    """swan_qa_studies: synthetic correctness bench."""
    return _finite_blob(swan_qa_studies.bench_swan_qa_studies(seed))
