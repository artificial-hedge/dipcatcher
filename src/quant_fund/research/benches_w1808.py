"""Wave-1808 bench adapters: phrygian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    agdistis2_qa_studies,
    attis2_qa_studies,
    cybele2_qa_studies,
    men2_qa_studies,
    papas2_qa_studies,
    sabazios2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18080


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_agdistis2_qa_studies_family(seed: int = _SEED + 0):
    """agdistis2_qa_studies: synthetic correctness bench."""
    return _finite_blob(agdistis2_qa_studies.bench_agdistis2_qa_studies(seed))


def bench_attis2_qa_studies_family(seed: int = _SEED + 1):
    """attis2_qa_studies: synthetic correctness bench."""
    return _finite_blob(attis2_qa_studies.bench_attis2_qa_studies(seed))


def bench_cybele2_qa_studies_family(seed: int = _SEED + 2):
    """cybele2_qa_studies: synthetic correctness bench."""
    return _finite_blob(cybele2_qa_studies.bench_cybele2_qa_studies(seed))


def bench_men2_qa_studies_family(seed: int = _SEED + 3):
    """men2_qa_studies: synthetic correctness bench."""
    return _finite_blob(men2_qa_studies.bench_men2_qa_studies(seed))


def bench_papas2_qa_studies_family(seed: int = _SEED + 4):
    """papas2_qa_studies: synthetic correctness bench."""
    return _finite_blob(papas2_qa_studies.bench_papas2_qa_studies(seed))


def bench_sabazios2_qa_studies_family(seed: int = _SEED + 5):
    """sabazios2_qa_studies: synthetic correctness bench."""
    return _finite_blob(sabazios2_qa_studies.bench_sabazios2_qa_studies(seed))
