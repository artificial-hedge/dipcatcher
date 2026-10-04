"""Wave-1729 bench adapters: baltic-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    austra_qa_studies,
    jumis_qa_studies,
    laima_qa_studies,
    laume_qa_studies,
    perkunas_qa_studies,
    zemyna_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17290


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_austra_qa_studies_family(seed: int = _SEED + 0):
    """austra_qa_studies: synthetic correctness bench."""
    return _finite_blob(austra_qa_studies.bench_austra_qa_studies(seed))


def bench_jumis_qa_studies_family(seed: int = _SEED + 1):
    """jumis_qa_studies: synthetic correctness bench."""
    return _finite_blob(jumis_qa_studies.bench_jumis_qa_studies(seed))


def bench_laima_qa_studies_family(seed: int = _SEED + 2):
    """laima_qa_studies: synthetic correctness bench."""
    return _finite_blob(laima_qa_studies.bench_laima_qa_studies(seed))


def bench_laume_qa_studies_family(seed: int = _SEED + 3):
    """laume_qa_studies: synthetic correctness bench."""
    return _finite_blob(laume_qa_studies.bench_laume_qa_studies(seed))


def bench_perkunas_qa_studies_family(seed: int = _SEED + 4):
    """perkunas_qa_studies: synthetic correctness bench."""
    return _finite_blob(perkunas_qa_studies.bench_perkunas_qa_studies(seed))


def bench_zemyna_qa_studies_family(seed: int = _SEED + 5):
    """zemyna_qa_studies: synthetic correctness bench."""
    return _finite_blob(zemyna_qa_studies.bench_zemyna_qa_studies(seed))
