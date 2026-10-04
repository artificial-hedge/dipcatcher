"""Wave-1459 bench adapters: arctic canon (SYNTHETIC only)."""

from quant_fund.models import (
    arctic_fox_qa_studies,
    caribou_qa_studies,
    musk_ox_qa_studies,
    penguin_qa_studies,
    polar_bear_qa_studies,
    reindeer_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14590


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_arctic_fox_qa_studies_family(seed: int = _SEED + 0):
    """arctic_fox_qa_studies: synthetic correctness bench."""
    return _finite_blob(arctic_fox_qa_studies.bench_arctic_fox_qa_studies(seed))


def bench_caribou_qa_studies_family(seed: int = _SEED + 1):
    """caribou_qa_studies: synthetic correctness bench."""
    return _finite_blob(caribou_qa_studies.bench_caribou_qa_studies(seed))


def bench_musk_ox_qa_studies_family(seed: int = _SEED + 2):
    """musk_ox_qa_studies: synthetic correctness bench."""
    return _finite_blob(musk_ox_qa_studies.bench_musk_ox_qa_studies(seed))


def bench_penguin_qa_studies_family(seed: int = _SEED + 3):
    """penguin_qa_studies: synthetic correctness bench."""
    return _finite_blob(penguin_qa_studies.bench_penguin_qa_studies(seed))


def bench_polar_bear_qa_studies_family(seed: int = _SEED + 4):
    """polar_bear_qa_studies: synthetic correctness bench."""
    return _finite_blob(polar_bear_qa_studies.bench_polar_bear_qa_studies(seed))


def bench_reindeer_qa_studies_family(seed: int = _SEED + 5):
    """reindeer_qa_studies: synthetic correctness bench."""
    return _finite_blob(reindeer_qa_studies.bench_reindeer_qa_studies(seed))
