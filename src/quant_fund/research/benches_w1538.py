"""Wave-1538 bench adapters: wetland canon (SYNTHETIC only)."""

from quant_fund.models import (
    crowned_crane_qa_studies,
    demoiselle_qa_studies,
    finfoot_qa_studies,
    limpkin_qa_studies,
    trumpeter_qa_studies,
    whooping_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15380


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_crowned_crane_qa_studies_family(seed: int = _SEED + 0):
    """crowned_crane_qa_studies: synthetic correctness bench."""
    return _finite_blob(crowned_crane_qa_studies.bench_crowned_crane_qa_studies(seed))


def bench_demoiselle_qa_studies_family(seed: int = _SEED + 1):
    """demoiselle_qa_studies: synthetic correctness bench."""
    return _finite_blob(demoiselle_qa_studies.bench_demoiselle_qa_studies(seed))


def bench_finfoot_qa_studies_family(seed: int = _SEED + 2):
    """finfoot_qa_studies: synthetic correctness bench."""
    return _finite_blob(finfoot_qa_studies.bench_finfoot_qa_studies(seed))


def bench_limpkin_qa_studies_family(seed: int = _SEED + 3):
    """limpkin_qa_studies: synthetic correctness bench."""
    return _finite_blob(limpkin_qa_studies.bench_limpkin_qa_studies(seed))


def bench_trumpeter_qa_studies_family(seed: int = _SEED + 4):
    """trumpeter_qa_studies: synthetic correctness bench."""
    return _finite_blob(trumpeter_qa_studies.bench_trumpeter_qa_studies(seed))


def bench_whooping_qa_studies_family(seed: int = _SEED + 5):
    """whooping_qa_studies: synthetic correctness bench."""
    return _finite_blob(whooping_qa_studies.bench_whooping_qa_studies(seed))
