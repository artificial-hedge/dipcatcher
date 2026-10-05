"""Wave-1942 bench adapters: vietnamese-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    co_hon_qa_studies,
    hon_ma_qa_studies,
    ngu_tinh_qa_studies,
    quy_am_qa_studies,
    tinh_linh_qa_studies,
    yeu_quai_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19420


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_co_hon_qa_studies_family(seed: int = _SEED + 0):
    """co_hon_qa_studies: synthetic correctness bench."""
    return _finite_blob(co_hon_qa_studies.bench_co_hon_qa_studies(seed))


def bench_hon_ma_qa_studies_family(seed: int = _SEED + 1):
    """hon_ma_qa_studies: synthetic correctness bench."""
    return _finite_blob(hon_ma_qa_studies.bench_hon_ma_qa_studies(seed))


def bench_ngu_tinh_qa_studies_family(seed: int = _SEED + 2):
    """ngu_tinh_qa_studies: synthetic correctness bench."""
    return _finite_blob(ngu_tinh_qa_studies.bench_ngu_tinh_qa_studies(seed))


def bench_quy_am_qa_studies_family(seed: int = _SEED + 3):
    """quy_am_qa_studies: synthetic correctness bench."""
    return _finite_blob(quy_am_qa_studies.bench_quy_am_qa_studies(seed))


def bench_tinh_linh_qa_studies_family(seed: int = _SEED + 4):
    """tinh_linh_qa_studies: synthetic correctness bench."""
    return _finite_blob(tinh_linh_qa_studies.bench_tinh_linh_qa_studies(seed))


def bench_yeu_quai_qa_studies_family(seed: int = _SEED + 5):
    """yeu_quai_qa_studies: synthetic correctness bench."""
    return _finite_blob(yeu_quai_qa_studies.bench_yeu_quai_qa_studies(seed))
