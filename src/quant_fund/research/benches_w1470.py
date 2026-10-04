"""Wave-1470 bench adapters: evergreen canon (SYNTHETIC only)."""

from quant_fund.models import (
    aspen_qa_studies,
    fir_qa_studies,
    holly_qa_studies,
    juniper_qa_studies,
    redwood_qa_studies,
    sequoia_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14700


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aspen_qa_studies_family(seed: int = _SEED + 0):
    """aspen_qa_studies: synthetic correctness bench."""
    return _finite_blob(aspen_qa_studies.bench_aspen_qa_studies(seed))


def bench_fir_qa_studies_family(seed: int = _SEED + 1):
    """fir_qa_studies: synthetic correctness bench."""
    return _finite_blob(fir_qa_studies.bench_fir_qa_studies(seed))


def bench_holly_qa_studies_family(seed: int = _SEED + 2):
    """holly_qa_studies: synthetic correctness bench."""
    return _finite_blob(holly_qa_studies.bench_holly_qa_studies(seed))


def bench_juniper_qa_studies_family(seed: int = _SEED + 3):
    """juniper_qa_studies: synthetic correctness bench."""
    return _finite_blob(juniper_qa_studies.bench_juniper_qa_studies(seed))


def bench_redwood_qa_studies_family(seed: int = _SEED + 4):
    """redwood_qa_studies: synthetic correctness bench."""
    return _finite_blob(redwood_qa_studies.bench_redwood_qa_studies(seed))


def bench_sequoia_qa_studies_family(seed: int = _SEED + 5):
    """sequoia_qa_studies: synthetic correctness bench."""
    return _finite_blob(sequoia_qa_studies.bench_sequoia_qa_studies(seed))
