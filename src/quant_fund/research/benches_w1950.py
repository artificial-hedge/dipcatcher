"""Wave-1950 bench adapters: goetic-assembly canon (SYNTHETIC only)."""

from quant_fund.models import (
    beleth_qa_studies,
    botis_qa_studies,
    eligos_qa_studies,
    leraje_qa_studies,
    sitri_qa_studies,
    zepar_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19500


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_beleth_qa_studies_family(seed: int = _SEED + 0):
    """beleth_qa_studies: synthetic correctness bench."""
    return _finite_blob(beleth_qa_studies.bench_beleth_qa_studies(seed))


def bench_botis_qa_studies_family(seed: int = _SEED + 1):
    """botis_qa_studies: synthetic correctness bench."""
    return _finite_blob(botis_qa_studies.bench_botis_qa_studies(seed))


def bench_eligos_qa_studies_family(seed: int = _SEED + 2):
    """eligos_qa_studies: synthetic correctness bench."""
    return _finite_blob(eligos_qa_studies.bench_eligos_qa_studies(seed))


def bench_leraje_qa_studies_family(seed: int = _SEED + 3):
    """leraje_qa_studies: synthetic correctness bench."""
    return _finite_blob(leraje_qa_studies.bench_leraje_qa_studies(seed))


def bench_sitri_qa_studies_family(seed: int = _SEED + 4):
    """sitri_qa_studies: synthetic correctness bench."""
    return _finite_blob(sitri_qa_studies.bench_sitri_qa_studies(seed))


def bench_zepar_qa_studies_family(seed: int = _SEED + 5):
    """zepar_qa_studies: synthetic correctness bench."""
    return _finite_blob(zepar_qa_studies.bench_zepar_qa_studies(seed))
