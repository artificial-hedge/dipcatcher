"""Wave-1591 bench adapters: new-world-monkey canon (SYNTHETIC only)."""

from quant_fund.models import (
    capuchin_qa_studies,
    saki_qa_studies,
    squirrel_monkey_qa_studies,
    titi_qa_studies,
    uakari_qa_studies,
    woolly_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15910


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_capuchin_qa_studies_family(seed: int = _SEED + 0):
    """capuchin_qa_studies: synthetic correctness bench."""
    return _finite_blob(capuchin_qa_studies.bench_capuchin_qa_studies(seed))


def bench_saki_qa_studies_family(seed: int = _SEED + 1):
    """saki_qa_studies: synthetic correctness bench."""
    return _finite_blob(saki_qa_studies.bench_saki_qa_studies(seed))


def bench_squirrel_monkey_qa_studies_family(seed: int = _SEED + 2):
    """squirrel_monkey_qa_studies: synthetic correctness bench."""
    return _finite_blob(squirrel_monkey_qa_studies.bench_squirrel_monkey_qa_studies(seed))


def bench_titi_qa_studies_family(seed: int = _SEED + 3):
    """titi_qa_studies: synthetic correctness bench."""
    return _finite_blob(titi_qa_studies.bench_titi_qa_studies(seed))


def bench_uakari_qa_studies_family(seed: int = _SEED + 4):
    """uakari_qa_studies: synthetic correctness bench."""
    return _finite_blob(uakari_qa_studies.bench_uakari_qa_studies(seed))


def bench_woolly_qa_studies_family(seed: int = _SEED + 5):
    """woolly_qa_studies: synthetic correctness bench."""
    return _finite_blob(woolly_qa_studies.bench_woolly_qa_studies(seed))
