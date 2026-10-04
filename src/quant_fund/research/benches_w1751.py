"""Wave-1751 bench adapters: egyptian-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    anubis_qa_studies,
    bastet_qa_studies,
    khonsu_qa_studies,
    min_qa_studies,
    neith_qa_studies,
    sobek_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17510


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_anubis_qa_studies_family(seed: int = _SEED + 0):
    """anubis_qa_studies: synthetic correctness bench."""
    return _finite_blob(anubis_qa_studies.bench_anubis_qa_studies(seed))


def bench_bastet_qa_studies_family(seed: int = _SEED + 1):
    """bastet_qa_studies: synthetic correctness bench."""
    return _finite_blob(bastet_qa_studies.bench_bastet_qa_studies(seed))


def bench_khonsu_qa_studies_family(seed: int = _SEED + 2):
    """khonsu_qa_studies: synthetic correctness bench."""
    return _finite_blob(khonsu_qa_studies.bench_khonsu_qa_studies(seed))


def bench_min_qa_studies_family(seed: int = _SEED + 3):
    """min_qa_studies: synthetic correctness bench."""
    return _finite_blob(min_qa_studies.bench_min_qa_studies(seed))


def bench_neith_qa_studies_family(seed: int = _SEED + 4):
    """neith_qa_studies: synthetic correctness bench."""
    return _finite_blob(neith_qa_studies.bench_neith_qa_studies(seed))


def bench_sobek_qa_studies_family(seed: int = _SEED + 5):
    """sobek_qa_studies: synthetic correctness bench."""
    return _finite_blob(sobek_qa_studies.bench_sobek_qa_studies(seed))
