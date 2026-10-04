"""Wave-1506 bench adapters: duck canon (SYNTHETIC only)."""

from quant_fund.models import (
    bufflehead_qa_studies,
    canvasback_qa_studies,
    eider_qa_studies,
    mallard_qa_studies,
    merganser_qa_studies,
    scoter_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15060


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bufflehead_qa_studies_family(seed: int = _SEED + 0):
    """bufflehead_qa_studies: synthetic correctness bench."""
    return _finite_blob(bufflehead_qa_studies.bench_bufflehead_qa_studies(seed))


def bench_canvasback_qa_studies_family(seed: int = _SEED + 1):
    """canvasback_qa_studies: synthetic correctness bench."""
    return _finite_blob(canvasback_qa_studies.bench_canvasback_qa_studies(seed))


def bench_eider_qa_studies_family(seed: int = _SEED + 2):
    """eider_qa_studies: synthetic correctness bench."""
    return _finite_blob(eider_qa_studies.bench_eider_qa_studies(seed))


def bench_mallard_qa_studies_family(seed: int = _SEED + 3):
    """mallard_qa_studies: synthetic correctness bench."""
    return _finite_blob(mallard_qa_studies.bench_mallard_qa_studies(seed))


def bench_merganser_qa_studies_family(seed: int = _SEED + 4):
    """merganser_qa_studies: synthetic correctness bench."""
    return _finite_blob(merganser_qa_studies.bench_merganser_qa_studies(seed))


def bench_scoter_qa_studies_family(seed: int = _SEED + 5):
    """scoter_qa_studies: synthetic correctness bench."""
    return _finite_blob(scoter_qa_studies.bench_scoter_qa_studies(seed))
