"""Wave-1489 bench adapters: bloom canon (SYNTHETIC only)."""

from quant_fund.models import (
    clover_qa_studies,
    heather_qa_studies,
    lavender_qa_studies,
    lilac_qa_studies,
    marigold_qa_studies,
    primrose_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14890


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_clover_qa_studies_family(seed: int = _SEED + 0):
    """clover_qa_studies: synthetic correctness bench."""
    return _finite_blob(clover_qa_studies.bench_clover_qa_studies(seed))


def bench_heather_qa_studies_family(seed: int = _SEED + 1):
    """heather_qa_studies: synthetic correctness bench."""
    return _finite_blob(heather_qa_studies.bench_heather_qa_studies(seed))


def bench_lavender_qa_studies_family(seed: int = _SEED + 2):
    """lavender_qa_studies: synthetic correctness bench."""
    return _finite_blob(lavender_qa_studies.bench_lavender_qa_studies(seed))


def bench_lilac_qa_studies_family(seed: int = _SEED + 3):
    """lilac_qa_studies: synthetic correctness bench."""
    return _finite_blob(lilac_qa_studies.bench_lilac_qa_studies(seed))


def bench_marigold_qa_studies_family(seed: int = _SEED + 4):
    """marigold_qa_studies: synthetic correctness bench."""
    return _finite_blob(marigold_qa_studies.bench_marigold_qa_studies(seed))


def bench_primrose_qa_studies_family(seed: int = _SEED + 5):
    """primrose_qa_studies: synthetic correctness bench."""
    return _finite_blob(primrose_qa_studies.bench_primrose_qa_studies(seed))
