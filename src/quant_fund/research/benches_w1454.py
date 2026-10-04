"""Wave-1454 bench adapters: insect-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    aphid_qa_studies,
    hornet_qa_studies,
    locust_qa_studies,
    mosquito_qa_studies,
    scarab_qa_studies,
    termite_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14540


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aphid_qa_studies_family(seed: int = _SEED + 0):
    """aphid_qa_studies: synthetic correctness bench."""
    return _finite_blob(aphid_qa_studies.bench_aphid_qa_studies(seed))


def bench_hornet_qa_studies_family(seed: int = _SEED + 1):
    """hornet_qa_studies: synthetic correctness bench."""
    return _finite_blob(hornet_qa_studies.bench_hornet_qa_studies(seed))


def bench_locust_qa_studies_family(seed: int = _SEED + 2):
    """locust_qa_studies: synthetic correctness bench."""
    return _finite_blob(locust_qa_studies.bench_locust_qa_studies(seed))


def bench_mosquito_qa_studies_family(seed: int = _SEED + 3):
    """mosquito_qa_studies: synthetic correctness bench."""
    return _finite_blob(mosquito_qa_studies.bench_mosquito_qa_studies(seed))


def bench_scarab_qa_studies_family(seed: int = _SEED + 4):
    """scarab_qa_studies: synthetic correctness bench."""
    return _finite_blob(scarab_qa_studies.bench_scarab_qa_studies(seed))


def bench_termite_qa_studies_family(seed: int = _SEED + 5):
    """termite_qa_studies: synthetic correctness bench."""
    return _finite_blob(termite_qa_studies.bench_termite_qa_studies(seed))
