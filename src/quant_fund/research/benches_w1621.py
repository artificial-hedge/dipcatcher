"""Wave-1621 bench adapters: cave-dwelling canon (SYNTHETIC only)."""

from quant_fund.models import (
    cave_beetle_qa_studies,
    cave_cricket_qa_studies,
    cave_fish_qa_studies,
    mudpuppy_qa_studies,
    olm_qa_studies,
    troglobite_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16210


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cave_beetle_qa_studies_family(seed: int = _SEED + 0):
    """cave_beetle_qa_studies: synthetic correctness bench."""
    return _finite_blob(cave_beetle_qa_studies.bench_cave_beetle_qa_studies(seed))


def bench_cave_cricket_qa_studies_family(seed: int = _SEED + 1):
    """cave_cricket_qa_studies: synthetic correctness bench."""
    return _finite_blob(cave_cricket_qa_studies.bench_cave_cricket_qa_studies(seed))


def bench_cave_fish_qa_studies_family(seed: int = _SEED + 2):
    """cave_fish_qa_studies: synthetic correctness bench."""
    return _finite_blob(cave_fish_qa_studies.bench_cave_fish_qa_studies(seed))


def bench_mudpuppy_qa_studies_family(seed: int = _SEED + 3):
    """mudpuppy_qa_studies: synthetic correctness bench."""
    return _finite_blob(mudpuppy_qa_studies.bench_mudpuppy_qa_studies(seed))


def bench_olm_qa_studies_family(seed: int = _SEED + 4):
    """olm_qa_studies: synthetic correctness bench."""
    return _finite_blob(olm_qa_studies.bench_olm_qa_studies(seed))


def bench_troglobite_qa_studies_family(seed: int = _SEED + 5):
    """troglobite_qa_studies: synthetic correctness bench."""
    return _finite_blob(troglobite_qa_studies.bench_troglobite_qa_studies(seed))
