"""Wave-1791 bench adapters: norse-myth-12 canon (SYNTHETIC only)."""

from quant_fund.models import (
    frey_qa_studies,
    freya2_qa_studies,
    magni_qa_studies,
    modi_qa_studies,
    njord_qa_studies,
    tyr2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17910


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_frey_qa_studies_family(seed: int = _SEED + 0):
    """frey_qa_studies: synthetic correctness bench."""
    return _finite_blob(frey_qa_studies.bench_frey_qa_studies(seed))


def bench_freya2_qa_studies_family(seed: int = _SEED + 1):
    """freya2_qa_studies: synthetic correctness bench."""
    return _finite_blob(freya2_qa_studies.bench_freya2_qa_studies(seed))


def bench_magni_qa_studies_family(seed: int = _SEED + 2):
    """magni_qa_studies: synthetic correctness bench."""
    return _finite_blob(magni_qa_studies.bench_magni_qa_studies(seed))


def bench_modi_qa_studies_family(seed: int = _SEED + 3):
    """modi_qa_studies: synthetic correctness bench."""
    return _finite_blob(modi_qa_studies.bench_modi_qa_studies(seed))


def bench_njord_qa_studies_family(seed: int = _SEED + 4):
    """njord_qa_studies: synthetic correctness bench."""
    return _finite_blob(njord_qa_studies.bench_njord_qa_studies(seed))


def bench_tyr2_qa_studies_family(seed: int = _SEED + 5):
    """tyr2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tyr2_qa_studies.bench_tyr2_qa_studies(seed))
