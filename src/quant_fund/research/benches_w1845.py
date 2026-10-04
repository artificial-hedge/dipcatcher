"""Wave-1845 bench adapters: arabian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    allat_qa_studies,
    dushara_qa_studies,
    hubal_qa_studies,
    manat_qa_studies,
    uzza_qa_studies,
    wadd_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18450


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_allat_qa_studies_family(seed: int = _SEED + 0):
    """allat_qa_studies: synthetic correctness bench."""
    return _finite_blob(allat_qa_studies.bench_allat_qa_studies(seed))


def bench_dushara_qa_studies_family(seed: int = _SEED + 1):
    """dushara_qa_studies: synthetic correctness bench."""
    return _finite_blob(dushara_qa_studies.bench_dushara_qa_studies(seed))


def bench_hubal_qa_studies_family(seed: int = _SEED + 2):
    """hubal_qa_studies: synthetic correctness bench."""
    return _finite_blob(hubal_qa_studies.bench_hubal_qa_studies(seed))


def bench_manat_qa_studies_family(seed: int = _SEED + 3):
    """manat_qa_studies: synthetic correctness bench."""
    return _finite_blob(manat_qa_studies.bench_manat_qa_studies(seed))


def bench_uzza_qa_studies_family(seed: int = _SEED + 4):
    """uzza_qa_studies: synthetic correctness bench."""
    return _finite_blob(uzza_qa_studies.bench_uzza_qa_studies(seed))


def bench_wadd_qa_studies_family(seed: int = _SEED + 5):
    """wadd_qa_studies: synthetic correctness bench."""
    return _finite_blob(wadd_qa_studies.bench_wadd_qa_studies(seed))
