"""Wave-1572 bench adapters: wildflower canon (SYNTHETIC only)."""

from quant_fund.models import (
    aster_qa_studies,
    bluebell_qa_studies,
    buttercup_qa_studies,
    columbine_qa_studies,
    cornflower_qa_studies,
    lupine_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15720


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aster_qa_studies_family(seed: int = _SEED + 0):
    """aster_qa_studies: synthetic correctness bench."""
    return _finite_blob(aster_qa_studies.bench_aster_qa_studies(seed))


def bench_bluebell_qa_studies_family(seed: int = _SEED + 1):
    """bluebell_qa_studies: synthetic correctness bench."""
    return _finite_blob(bluebell_qa_studies.bench_bluebell_qa_studies(seed))


def bench_buttercup_qa_studies_family(seed: int = _SEED + 2):
    """buttercup_qa_studies: synthetic correctness bench."""
    return _finite_blob(buttercup_qa_studies.bench_buttercup_qa_studies(seed))


def bench_columbine_qa_studies_family(seed: int = _SEED + 3):
    """columbine_qa_studies: synthetic correctness bench."""
    return _finite_blob(columbine_qa_studies.bench_columbine_qa_studies(seed))


def bench_cornflower_qa_studies_family(seed: int = _SEED + 4):
    """cornflower_qa_studies: synthetic correctness bench."""
    return _finite_blob(cornflower_qa_studies.bench_cornflower_qa_studies(seed))


def bench_lupine_qa_studies_family(seed: int = _SEED + 5):
    """lupine_qa_studies: synthetic correctness bench."""
    return _finite_blob(lupine_qa_studies.bench_lupine_qa_studies(seed))
