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
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
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
