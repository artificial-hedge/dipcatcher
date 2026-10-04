"""Wave-1805 bench adapters: slavic-myth-5 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bereginia2_qa_studies,
    bogdan2_qa_studies,
    kupalo2_qa_studies,
    radegast2_qa_studies,
    rod2_qa_studies,
    ziva2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18050


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bereginia2_qa_studies_family(seed: int = _SEED + 0):
    """bereginia2_qa_studies: synthetic correctness bench."""
    return _finite_blob(bereginia2_qa_studies.bench_bereginia2_qa_studies(seed))


def bench_bogdan2_qa_studies_family(seed: int = _SEED + 1):
    """bogdan2_qa_studies: synthetic correctness bench."""
    return _finite_blob(bogdan2_qa_studies.bench_bogdan2_qa_studies(seed))


def bench_kupalo2_qa_studies_family(seed: int = _SEED + 2):
    """kupalo2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kupalo2_qa_studies.bench_kupalo2_qa_studies(seed))


def bench_radegast2_qa_studies_family(seed: int = _SEED + 3):
    """radegast2_qa_studies: synthetic correctness bench."""
    return _finite_blob(radegast2_qa_studies.bench_radegast2_qa_studies(seed))


def bench_rod2_qa_studies_family(seed: int = _SEED + 4):
    """rod2_qa_studies: synthetic correctness bench."""
    return _finite_blob(rod2_qa_studies.bench_rod2_qa_studies(seed))


def bench_ziva2_qa_studies_family(seed: int = _SEED + 5):
    """ziva2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ziva2_qa_studies.bench_ziva2_qa_studies(seed))
