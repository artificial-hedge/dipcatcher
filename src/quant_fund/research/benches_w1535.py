"""Wave-1535 bench adapters: aerialist canon (SYNTHETIC only)."""

from quant_fund.models import (
    martin_qa_studies,
    needletail_qa_studies,
    swallow_qa_studies,
    swift_qa_studies,
    swiftlet_qa_studies,
    treeswift_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15350


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_martin_qa_studies_family(seed: int = _SEED + 0):
    """martin_qa_studies: synthetic correctness bench."""
    return _finite_blob(martin_qa_studies.bench_martin_qa_studies(seed))


def bench_needletail_qa_studies_family(seed: int = _SEED + 1):
    """needletail_qa_studies: synthetic correctness bench."""
    return _finite_blob(needletail_qa_studies.bench_needletail_qa_studies(seed))


def bench_swallow_qa_studies_family(seed: int = _SEED + 2):
    """swallow_qa_studies: synthetic correctness bench."""
    return _finite_blob(swallow_qa_studies.bench_swallow_qa_studies(seed))


def bench_swift_qa_studies_family(seed: int = _SEED + 3):
    """swift_qa_studies: synthetic correctness bench."""
    return _finite_blob(swift_qa_studies.bench_swift_qa_studies(seed))


def bench_swiftlet_qa_studies_family(seed: int = _SEED + 4):
    """swiftlet_qa_studies: synthetic correctness bench."""
    return _finite_blob(swiftlet_qa_studies.bench_swiftlet_qa_studies(seed))


def bench_treeswift_qa_studies_family(seed: int = _SEED + 5):
    """treeswift_qa_studies: synthetic correctness bench."""
    return _finite_blob(treeswift_qa_studies.bench_treeswift_qa_studies(seed))
