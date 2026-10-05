"""Wave-1836 bench adapters: carian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    axom2_qa_studies,
    chrysaoreus2_qa_studies,
    hekate2_qa_studies,
    labrandeus2_qa_studies,
    panamara2_qa_studies,
    stratios2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18360


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_axom2_qa_studies_family(seed: int = _SEED + 0):
    """axom2_qa_studies: synthetic correctness bench."""
    return _finite_blob(axom2_qa_studies.bench_axom2_qa_studies(seed))


def bench_chrysaoreus2_qa_studies_family(seed: int = _SEED + 1):
    """chrysaoreus2_qa_studies: synthetic correctness bench."""
    return _finite_blob(chrysaoreus2_qa_studies.bench_chrysaoreus2_qa_studies(seed))


def bench_hekate2_qa_studies_family(seed: int = _SEED + 2):
    """hekate2_qa_studies: synthetic correctness bench."""
    return _finite_blob(hekate2_qa_studies.bench_hekate2_qa_studies(seed))


def bench_labrandeus2_qa_studies_family(seed: int = _SEED + 3):
    """labrandeus2_qa_studies: synthetic correctness bench."""
    return _finite_blob(labrandeus2_qa_studies.bench_labrandeus2_qa_studies(seed))


def bench_panamara2_qa_studies_family(seed: int = _SEED + 4):
    """panamara2_qa_studies: synthetic correctness bench."""
    return _finite_blob(panamara2_qa_studies.bench_panamara2_qa_studies(seed))


def bench_stratios2_qa_studies_family(seed: int = _SEED + 5):
    """stratios2_qa_studies: synthetic correctness bench."""
    return _finite_blob(stratios2_qa_studies.bench_stratios2_qa_studies(seed))
