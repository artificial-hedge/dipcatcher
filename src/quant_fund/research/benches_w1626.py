"""Wave-1626 bench adapters: lemur-region canon (SYNTHETIC only)."""

from quant_fund.models import (
    bondolo_qa_studies,
    madame_berthe_qa_studies,
    mittermeier_qa_studies,
    northern_qa_studies,
    southern_qa_studies,
    western_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16260


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bondolo_qa_studies_family(seed: int = _SEED + 0):
    """bondolo_qa_studies: synthetic correctness bench."""
    return _finite_blob(bondolo_qa_studies.bench_bondolo_qa_studies(seed))


def bench_madame_berthe_qa_studies_family(seed: int = _SEED + 1):
    """madame_berthe_qa_studies: synthetic correctness bench."""
    return _finite_blob(madame_berthe_qa_studies.bench_madame_berthe_qa_studies(seed))


def bench_mittermeier_qa_studies_family(seed: int = _SEED + 2):
    """mittermeier_qa_studies: synthetic correctness bench."""
    return _finite_blob(mittermeier_qa_studies.bench_mittermeier_qa_studies(seed))


def bench_northern_qa_studies_family(seed: int = _SEED + 3):
    """northern_qa_studies: synthetic correctness bench."""
    return _finite_blob(northern_qa_studies.bench_northern_qa_studies(seed))


def bench_southern_qa_studies_family(seed: int = _SEED + 4):
    """southern_qa_studies: synthetic correctness bench."""
    return _finite_blob(southern_qa_studies.bench_southern_qa_studies(seed))


def bench_western_qa_studies_family(seed: int = _SEED + 5):
    """western_qa_studies: synthetic correctness bench."""
    return _finite_blob(western_qa_studies.bench_western_qa_studies(seed))
