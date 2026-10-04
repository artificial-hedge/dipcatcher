"""Wave-1536 bench adapters: columbid canon (SYNTHETIC only)."""

from quant_fund.models import (
    collared_dove_qa_studies,
    dove_qa_studies,
    mourning_dove_qa_studies,
    pigeon_qa_studies,
    turtle_dove_qa_studies,
    woodpigeon_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15360


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_collared_dove_qa_studies_family(seed: int = _SEED + 0):
    """collared_dove_qa_studies: synthetic correctness bench."""
    return _finite_blob(collared_dove_qa_studies.bench_collared_dove_qa_studies(seed))


def bench_dove_qa_studies_family(seed: int = _SEED + 1):
    """dove_qa_studies: synthetic correctness bench."""
    return _finite_blob(dove_qa_studies.bench_dove_qa_studies(seed))


def bench_mourning_dove_qa_studies_family(seed: int = _SEED + 2):
    """mourning_dove_qa_studies: synthetic correctness bench."""
    return _finite_blob(mourning_dove_qa_studies.bench_mourning_dove_qa_studies(seed))


def bench_pigeon_qa_studies_family(seed: int = _SEED + 3):
    """pigeon_qa_studies: synthetic correctness bench."""
    return _finite_blob(pigeon_qa_studies.bench_pigeon_qa_studies(seed))


def bench_turtle_dove_qa_studies_family(seed: int = _SEED + 4):
    """turtle_dove_qa_studies: synthetic correctness bench."""
    return _finite_blob(turtle_dove_qa_studies.bench_turtle_dove_qa_studies(seed))


def bench_woodpigeon_qa_studies_family(seed: int = _SEED + 5):
    """woodpigeon_qa_studies: synthetic correctness bench."""
    return _finite_blob(woodpigeon_qa_studies.bench_woodpigeon_qa_studies(seed))
