"""Wave-1925 bench adapters: siberian-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    abaasy_qa_studies,
    chedipe_qa_studies,
    kus_qa_studies,
    kyys_qa_studies,
    oror_qa_studies,
    urgut_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19250


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_abaasy_qa_studies_family(seed: int = _SEED + 0):
    """abaasy_qa_studies: synthetic correctness bench."""
    return _finite_blob(abaasy_qa_studies.bench_abaasy_qa_studies(seed))


def bench_chedipe_qa_studies_family(seed: int = _SEED + 1):
    """chedipe_qa_studies: synthetic correctness bench."""
    return _finite_blob(chedipe_qa_studies.bench_chedipe_qa_studies(seed))


def bench_kus_qa_studies_family(seed: int = _SEED + 2):
    """kus_qa_studies: synthetic correctness bench."""
    return _finite_blob(kus_qa_studies.bench_kus_qa_studies(seed))


def bench_kyys_qa_studies_family(seed: int = _SEED + 3):
    """kyys_qa_studies: synthetic correctness bench."""
    return _finite_blob(kyys_qa_studies.bench_kyys_qa_studies(seed))


def bench_oror_qa_studies_family(seed: int = _SEED + 4):
    """oror_qa_studies: synthetic correctness bench."""
    return _finite_blob(oror_qa_studies.bench_oror_qa_studies(seed))


def bench_urgut_qa_studies_family(seed: int = _SEED + 5):
    """urgut_qa_studies: synthetic correctness bench."""
    return _finite_blob(urgut_qa_studies.bench_urgut_qa_studies(seed))
