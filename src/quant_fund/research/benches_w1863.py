"""Wave-1863 bench adapters: arthurian-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    balin_qa_studies,
    lamorak_qa_studies,
    lot_qa_studies,
    marhaus_qa_studies,
    pellinor_qa_studies,
    uther_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18630


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_balin_qa_studies_family(seed: int = _SEED + 0):
    """balin_qa_studies: synthetic correctness bench."""
    return _finite_blob(balin_qa_studies.bench_balin_qa_studies(seed))


def bench_lamorak_qa_studies_family(seed: int = _SEED + 1):
    """lamorak_qa_studies: synthetic correctness bench."""
    return _finite_blob(lamorak_qa_studies.bench_lamorak_qa_studies(seed))


def bench_lot_qa_studies_family(seed: int = _SEED + 2):
    """lot_qa_studies: synthetic correctness bench."""
    return _finite_blob(lot_qa_studies.bench_lot_qa_studies(seed))


def bench_marhaus_qa_studies_family(seed: int = _SEED + 3):
    """marhaus_qa_studies: synthetic correctness bench."""
    return _finite_blob(marhaus_qa_studies.bench_marhaus_qa_studies(seed))


def bench_pellinor_qa_studies_family(seed: int = _SEED + 4):
    """pellinor_qa_studies: synthetic correctness bench."""
    return _finite_blob(pellinor_qa_studies.bench_pellinor_qa_studies(seed))


def bench_uther_qa_studies_family(seed: int = _SEED + 5):
    """uther_qa_studies: synthetic correctness bench."""
    return _finite_blob(uther_qa_studies.bench_uther_qa_studies(seed))
