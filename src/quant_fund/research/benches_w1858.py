"""Wave-1858 bench adapters: lusitanian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    aracus_qa_studies,
    cosus_qa_studies,
    cronia_qa_studies,
    munidis_qa_studies,
    quangeio_qa_studies,
    reo_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18580


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aracus_qa_studies_family(seed: int = _SEED + 0):
    """aracus_qa_studies: synthetic correctness bench."""
    return _finite_blob(aracus_qa_studies.bench_aracus_qa_studies(seed))


def bench_cosus_qa_studies_family(seed: int = _SEED + 1):
    """cosus_qa_studies: synthetic correctness bench."""
    return _finite_blob(cosus_qa_studies.bench_cosus_qa_studies(seed))


def bench_cronia_qa_studies_family(seed: int = _SEED + 2):
    """cronia_qa_studies: synthetic correctness bench."""
    return _finite_blob(cronia_qa_studies.bench_cronia_qa_studies(seed))


def bench_munidis_qa_studies_family(seed: int = _SEED + 3):
    """munidis_qa_studies: synthetic correctness bench."""
    return _finite_blob(munidis_qa_studies.bench_munidis_qa_studies(seed))


def bench_quangeio_qa_studies_family(seed: int = _SEED + 4):
    """quangeio_qa_studies: synthetic correctness bench."""
    return _finite_blob(quangeio_qa_studies.bench_quangeio_qa_studies(seed))


def bench_reo_qa_studies_family(seed: int = _SEED + 5):
    """reo_qa_studies: synthetic correctness bench."""
    return _finite_blob(reo_qa_studies.bench_reo_qa_studies(seed))
