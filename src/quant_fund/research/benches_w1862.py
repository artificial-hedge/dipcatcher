"""Wave-1862 bench adapters: arthurian-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    agravaine_qa_studies,
    isolde_qa_studies,
    kay_qa_studies,
    lyonesse_qa_studies,
    mark_cornwall_qa_studies,
    mordred_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18620


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_agravaine_qa_studies_family(seed: int = _SEED + 0):
    """agravaine_qa_studies: synthetic correctness bench."""
    return _finite_blob(agravaine_qa_studies.bench_agravaine_qa_studies(seed))


def bench_isolde_qa_studies_family(seed: int = _SEED + 1):
    """isolde_qa_studies: synthetic correctness bench."""
    return _finite_blob(isolde_qa_studies.bench_isolde_qa_studies(seed))


def bench_kay_qa_studies_family(seed: int = _SEED + 2):
    """kay_qa_studies: synthetic correctness bench."""
    return _finite_blob(kay_qa_studies.bench_kay_qa_studies(seed))


def bench_lyonesse_qa_studies_family(seed: int = _SEED + 3):
    """lyonesse_qa_studies: synthetic correctness bench."""
    return _finite_blob(lyonesse_qa_studies.bench_lyonesse_qa_studies(seed))


def bench_mark_cornwall_qa_studies_family(seed: int = _SEED + 4):
    """mark_cornwall_qa_studies: synthetic correctness bench."""
    return _finite_blob(mark_cornwall_qa_studies.bench_mark_cornwall_qa_studies(seed))


def bench_mordred_qa_studies_family(seed: int = _SEED + 5):
    """mordred_qa_studies: synthetic correctness bench."""
    return _finite_blob(mordred_qa_studies.bench_mordred_qa_studies(seed))
