"""Wave-1800 bench adapters: roman-hero canon (SYNTHETIC only)."""

from quant_fund.models import (
    aeneas2_qa_studies,
    evander2_qa_studies,
    lavinia2_qa_studies,
    remus2_qa_studies,
    romulus2_qa_studies,
    turnus2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18000


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aeneas2_qa_studies_family(seed: int = _SEED + 0):
    """aeneas2_qa_studies: synthetic correctness bench."""
    return _finite_blob(aeneas2_qa_studies.bench_aeneas2_qa_studies(seed))


def bench_evander2_qa_studies_family(seed: int = _SEED + 1):
    """evander2_qa_studies: synthetic correctness bench."""
    return _finite_blob(evander2_qa_studies.bench_evander2_qa_studies(seed))


def bench_lavinia2_qa_studies_family(seed: int = _SEED + 2):
    """lavinia2_qa_studies: synthetic correctness bench."""
    return _finite_blob(lavinia2_qa_studies.bench_lavinia2_qa_studies(seed))


def bench_remus2_qa_studies_family(seed: int = _SEED + 3):
    """remus2_qa_studies: synthetic correctness bench."""
    return _finite_blob(remus2_qa_studies.bench_remus2_qa_studies(seed))


def bench_romulus2_qa_studies_family(seed: int = _SEED + 4):
    """romulus2_qa_studies: synthetic correctness bench."""
    return _finite_blob(romulus2_qa_studies.bench_romulus2_qa_studies(seed))


def bench_turnus2_qa_studies_family(seed: int = _SEED + 5):
    """turnus2_qa_studies: synthetic correctness bench."""
    return _finite_blob(turnus2_qa_studies.bench_turnus2_qa_studies(seed))
