"""Wave-1547 bench adapters: columbid-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    crowned_pigeon_qa_studies,
    cuckoo_dove_qa_studies,
    emerald_dove_qa_studies,
    fruit_dove_qa_studies,
    ground_dove_qa_studies,
    quail_dove_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15470


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_crowned_pigeon_qa_studies_family(seed: int = _SEED + 0):
    """crowned_pigeon_qa_studies: synthetic correctness bench."""
    return _finite_blob(crowned_pigeon_qa_studies.bench_crowned_pigeon_qa_studies(seed))


def bench_cuckoo_dove_qa_studies_family(seed: int = _SEED + 1):
    """cuckoo_dove_qa_studies: synthetic correctness bench."""
    return _finite_blob(cuckoo_dove_qa_studies.bench_cuckoo_dove_qa_studies(seed))


def bench_emerald_dove_qa_studies_family(seed: int = _SEED + 2):
    """emerald_dove_qa_studies: synthetic correctness bench."""
    return _finite_blob(emerald_dove_qa_studies.bench_emerald_dove_qa_studies(seed))


def bench_fruit_dove_qa_studies_family(seed: int = _SEED + 3):
    """fruit_dove_qa_studies: synthetic correctness bench."""
    return _finite_blob(fruit_dove_qa_studies.bench_fruit_dove_qa_studies(seed))


def bench_ground_dove_qa_studies_family(seed: int = _SEED + 4):
    """ground_dove_qa_studies: synthetic correctness bench."""
    return _finite_blob(ground_dove_qa_studies.bench_ground_dove_qa_studies(seed))


def bench_quail_dove_qa_studies_family(seed: int = _SEED + 5):
    """quail_dove_qa_studies: synthetic correctness bench."""
    return _finite_blob(quail_dove_qa_studies.bench_quail_dove_qa_studies(seed))
