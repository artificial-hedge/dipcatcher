"""Wave-1524 bench adapters: sedge canon (SYNTHETIC only)."""

from quant_fund.models import (
    bulrush_qa_studies,
    carex_qa_studies,
    cattail_qa_studies,
    cottongrass_qa_studies,
    reed_qa_studies,
    rush_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15240


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bulrush_qa_studies_family(seed: int = _SEED + 0):
    """bulrush_qa_studies: synthetic correctness bench."""
    return _finite_blob(bulrush_qa_studies.bench_bulrush_qa_studies(seed))


def bench_carex_qa_studies_family(seed: int = _SEED + 1):
    """carex_qa_studies: synthetic correctness bench."""
    return _finite_blob(carex_qa_studies.bench_carex_qa_studies(seed))


def bench_cattail_qa_studies_family(seed: int = _SEED + 2):
    """cattail_qa_studies: synthetic correctness bench."""
    return _finite_blob(cattail_qa_studies.bench_cattail_qa_studies(seed))


def bench_cottongrass_qa_studies_family(seed: int = _SEED + 3):
    """cottongrass_qa_studies: synthetic correctness bench."""
    return _finite_blob(cottongrass_qa_studies.bench_cottongrass_qa_studies(seed))


def bench_reed_qa_studies_family(seed: int = _SEED + 4):
    """reed_qa_studies: synthetic correctness bench."""
    return _finite_blob(reed_qa_studies.bench_reed_qa_studies(seed))


def bench_rush_qa_studies_family(seed: int = _SEED + 5):
    """rush_qa_studies: synthetic correctness bench."""
    return _finite_blob(rush_qa_studies.bench_rush_qa_studies(seed))
