"""Wave-1776 bench adapters: korean-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bari_qa_studies,
    kongjwi_qa_studies,
    ondal_qa_studies,
    pyonggang_qa_studies,
    samshin_qa_studies,
    shimchong_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17760


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bari_qa_studies_family(seed: int = _SEED + 0):
    """bari_qa_studies: synthetic correctness bench."""
    return _finite_blob(bari_qa_studies.bench_bari_qa_studies(seed))


def bench_kongjwi_qa_studies_family(seed: int = _SEED + 1):
    """kongjwi_qa_studies: synthetic correctness bench."""
    return _finite_blob(kongjwi_qa_studies.bench_kongjwi_qa_studies(seed))


def bench_ondal_qa_studies_family(seed: int = _SEED + 2):
    """ondal_qa_studies: synthetic correctness bench."""
    return _finite_blob(ondal_qa_studies.bench_ondal_qa_studies(seed))


def bench_pyonggang_qa_studies_family(seed: int = _SEED + 3):
    """pyonggang_qa_studies: synthetic correctness bench."""
    return _finite_blob(pyonggang_qa_studies.bench_pyonggang_qa_studies(seed))


def bench_samshin_qa_studies_family(seed: int = _SEED + 4):
    """samshin_qa_studies: synthetic correctness bench."""
    return _finite_blob(samshin_qa_studies.bench_samshin_qa_studies(seed))


def bench_shimchong_qa_studies_family(seed: int = _SEED + 5):
    """shimchong_qa_studies: synthetic correctness bench."""
    return _finite_blob(shimchong_qa_studies.bench_shimchong_qa_studies(seed))
