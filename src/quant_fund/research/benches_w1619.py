"""Wave-1619 bench adapters: abyssal-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    blobfish_qa_studies,
    dragonfish_qa_studies,
    dumbo_qa_studies,
    fangtooth_qa_studies,
    gulper_qa_studies,
    tripodfish_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16190


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_blobfish_qa_studies_family(seed: int = _SEED + 0):
    """blobfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(blobfish_qa_studies.bench_blobfish_qa_studies(seed))


def bench_dragonfish_qa_studies_family(seed: int = _SEED + 1):
    """dragonfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(dragonfish_qa_studies.bench_dragonfish_qa_studies(seed))


def bench_dumbo_qa_studies_family(seed: int = _SEED + 2):
    """dumbo_qa_studies: synthetic correctness bench."""
    return _finite_blob(dumbo_qa_studies.bench_dumbo_qa_studies(seed))


def bench_fangtooth_qa_studies_family(seed: int = _SEED + 3):
    """fangtooth_qa_studies: synthetic correctness bench."""
    return _finite_blob(fangtooth_qa_studies.bench_fangtooth_qa_studies(seed))


def bench_gulper_qa_studies_family(seed: int = _SEED + 4):
    """gulper_qa_studies: synthetic correctness bench."""
    return _finite_blob(gulper_qa_studies.bench_gulper_qa_studies(seed))


def bench_tripodfish_qa_studies_family(seed: int = _SEED + 5):
    """tripodfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(tripodfish_qa_studies.bench_tripodfish_qa_studies(seed))
