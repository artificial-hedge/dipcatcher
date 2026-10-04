"""Wave-1905 bench adapters: yokai-9 canon (SYNTHETIC only)."""

from quant_fund.models import (
    betobeto_qa_studies,
    buruburu_qa_studies,
    hyakume_qa_studies,
    shachihoko_qa_studies,
    uwan_qa_studies,
    waira_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19050


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_betobeto_qa_studies_family(seed: int = _SEED + 0):
    """betobeto_qa_studies: synthetic correctness bench."""
    return _finite_blob(betobeto_qa_studies.bench_betobeto_qa_studies(seed))


def bench_buruburu_qa_studies_family(seed: int = _SEED + 1):
    """buruburu_qa_studies: synthetic correctness bench."""
    return _finite_blob(buruburu_qa_studies.bench_buruburu_qa_studies(seed))


def bench_hyakume_qa_studies_family(seed: int = _SEED + 2):
    """hyakume_qa_studies: synthetic correctness bench."""
    return _finite_blob(hyakume_qa_studies.bench_hyakume_qa_studies(seed))


def bench_shachihoko_qa_studies_family(seed: int = _SEED + 3):
    """shachihoko_qa_studies: synthetic correctness bench."""
    return _finite_blob(shachihoko_qa_studies.bench_shachihoko_qa_studies(seed))


def bench_uwan_qa_studies_family(seed: int = _SEED + 4):
    """uwan_qa_studies: synthetic correctness bench."""
    return _finite_blob(uwan_qa_studies.bench_uwan_qa_studies(seed))


def bench_waira_qa_studies_family(seed: int = _SEED + 5):
    """waira_qa_studies: synthetic correctness bench."""
    return _finite_blob(waira_qa_studies.bench_waira_qa_studies(seed))
