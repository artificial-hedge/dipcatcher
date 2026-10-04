"""Wave-1678 bench adapters: slavic-folk-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bannik_qa_studies,
    dvorovoi_qa_studies,
    mora_qa_studies,
    ovinnik_qa_studies,
    poludnica_qa_studies,
    vila_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16780


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bannik_qa_studies_family(seed: int = _SEED + 0):
    """bannik_qa_studies: synthetic correctness bench."""
    return _finite_blob(bannik_qa_studies.bench_bannik_qa_studies(seed))


def bench_dvorovoi_qa_studies_family(seed: int = _SEED + 1):
    """dvorovoi_qa_studies: synthetic correctness bench."""
    return _finite_blob(dvorovoi_qa_studies.bench_dvorovoi_qa_studies(seed))


def bench_mora_qa_studies_family(seed: int = _SEED + 2):
    """mora_qa_studies: synthetic correctness bench."""
    return _finite_blob(mora_qa_studies.bench_mora_qa_studies(seed))


def bench_ovinnik_qa_studies_family(seed: int = _SEED + 3):
    """ovinnik_qa_studies: synthetic correctness bench."""
    return _finite_blob(ovinnik_qa_studies.bench_ovinnik_qa_studies(seed))


def bench_poludnica_qa_studies_family(seed: int = _SEED + 4):
    """poludnica_qa_studies: synthetic correctness bench."""
    return _finite_blob(poludnica_qa_studies.bench_poludnica_qa_studies(seed))


def bench_vila_qa_studies_family(seed: int = _SEED + 5):
    """vila_qa_studies: synthetic correctness bench."""
    return _finite_blob(vila_qa_studies.bench_vila_qa_studies(seed))
