"""Wave-1753 bench adapters: chinese-myth-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    changxi_qa_studies,
    chiyou_qa_studies,
    gonggong_qa_studies,
    xihe_qa_studies,
    yinglong_qa_studies,
    zhurong_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17530


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_changxi_qa_studies_family(seed: int = _SEED + 0):
    """changxi_qa_studies: synthetic correctness bench."""
    return _finite_blob(changxi_qa_studies.bench_changxi_qa_studies(seed))


def bench_chiyou_qa_studies_family(seed: int = _SEED + 1):
    """chiyou_qa_studies: synthetic correctness bench."""
    return _finite_blob(chiyou_qa_studies.bench_chiyou_qa_studies(seed))


def bench_gonggong_qa_studies_family(seed: int = _SEED + 2):
    """gonggong_qa_studies: synthetic correctness bench."""
    return _finite_blob(gonggong_qa_studies.bench_gonggong_qa_studies(seed))


def bench_xihe_qa_studies_family(seed: int = _SEED + 3):
    """xihe_qa_studies: synthetic correctness bench."""
    return _finite_blob(xihe_qa_studies.bench_xihe_qa_studies(seed))


def bench_yinglong_qa_studies_family(seed: int = _SEED + 4):
    """yinglong_qa_studies: synthetic correctness bench."""
    return _finite_blob(yinglong_qa_studies.bench_yinglong_qa_studies(seed))


def bench_zhurong_qa_studies_family(seed: int = _SEED + 5):
    """zhurong_qa_studies: synthetic correctness bench."""
    return _finite_blob(zhurong_qa_studies.bench_zhurong_qa_studies(seed))
