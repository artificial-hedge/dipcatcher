"""Wave-1803 bench adapters: celtic-myth-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    andrasta2_qa_studies,
    borvo2_qa_studies,
    epona2_qa_studies,
    etercuni2_qa_studies,
    maponos2_qa_studies,
    rosmerta2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18030


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_andrasta2_qa_studies_family(seed: int = _SEED + 0):
    """andrasta2_qa_studies: synthetic correctness bench."""
    return _finite_blob(andrasta2_qa_studies.bench_andrasta2_qa_studies(seed))


def bench_borvo2_qa_studies_family(seed: int = _SEED + 1):
    """borvo2_qa_studies: synthetic correctness bench."""
    return _finite_blob(borvo2_qa_studies.bench_borvo2_qa_studies(seed))


def bench_epona2_qa_studies_family(seed: int = _SEED + 2):
    """epona2_qa_studies: synthetic correctness bench."""
    return _finite_blob(epona2_qa_studies.bench_epona2_qa_studies(seed))


def bench_etercuni2_qa_studies_family(seed: int = _SEED + 3):
    """etercuni2_qa_studies: synthetic correctness bench."""
    return _finite_blob(etercuni2_qa_studies.bench_etercuni2_qa_studies(seed))


def bench_maponos2_qa_studies_family(seed: int = _SEED + 4):
    """maponos2_qa_studies: synthetic correctness bench."""
    return _finite_blob(maponos2_qa_studies.bench_maponos2_qa_studies(seed))


def bench_rosmerta2_qa_studies_family(seed: int = _SEED + 5):
    """rosmerta2_qa_studies: synthetic correctness bench."""
    return _finite_blob(rosmerta2_qa_studies.bench_rosmerta2_qa_studies(seed))
