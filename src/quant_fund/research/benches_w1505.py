"""Wave-1505 bench adapters: waterfowl canon (SYNTHETIC only)."""

from quant_fund.models import (
    gadwall_qa_studies,
    pintail_qa_studies,
    pochard_qa_studies,
    shoveler_qa_studies,
    teal_qa_studies,
    wigeon_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15050


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_gadwall_qa_studies_family(seed: int = _SEED + 0):
    """gadwall_qa_studies: synthetic correctness bench."""
    return _finite_blob(gadwall_qa_studies.bench_gadwall_qa_studies(seed))


def bench_pintail_qa_studies_family(seed: int = _SEED + 1):
    """pintail_qa_studies: synthetic correctness bench."""
    return _finite_blob(pintail_qa_studies.bench_pintail_qa_studies(seed))


def bench_pochard_qa_studies_family(seed: int = _SEED + 2):
    """pochard_qa_studies: synthetic correctness bench."""
    return _finite_blob(pochard_qa_studies.bench_pochard_qa_studies(seed))


def bench_shoveler_qa_studies_family(seed: int = _SEED + 3):
    """shoveler_qa_studies: synthetic correctness bench."""
    return _finite_blob(shoveler_qa_studies.bench_shoveler_qa_studies(seed))


def bench_teal_qa_studies_family(seed: int = _SEED + 4):
    """teal_qa_studies: synthetic correctness bench."""
    return _finite_blob(teal_qa_studies.bench_teal_qa_studies(seed))


def bench_wigeon_qa_studies_family(seed: int = _SEED + 5):
    """wigeon_qa_studies: synthetic correctness bench."""
    return _finite_blob(wigeon_qa_studies.bench_wigeon_qa_studies(seed))
