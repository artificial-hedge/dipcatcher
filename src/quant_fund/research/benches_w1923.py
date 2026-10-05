"""Wave-1923 bench adapters: chinese-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    baigujing_qa_studies,
    hanba_qa_studies,
    jiuying_qa_studies,
    nian_qa_studies,
    wuzhiqi_qa_studies,
    xiangliu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19230


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_baigujing_qa_studies_family(seed: int = _SEED + 0):
    """baigujing_qa_studies: synthetic correctness bench."""
    return _finite_blob(baigujing_qa_studies.bench_baigujing_qa_studies(seed))


def bench_hanba_qa_studies_family(seed: int = _SEED + 1):
    """hanba_qa_studies: synthetic correctness bench."""
    return _finite_blob(hanba_qa_studies.bench_hanba_qa_studies(seed))


def bench_jiuying_qa_studies_family(seed: int = _SEED + 2):
    """jiuying_qa_studies: synthetic correctness bench."""
    return _finite_blob(jiuying_qa_studies.bench_jiuying_qa_studies(seed))


def bench_nian_qa_studies_family(seed: int = _SEED + 3):
    """nian_qa_studies: synthetic correctness bench."""
    return _finite_blob(nian_qa_studies.bench_nian_qa_studies(seed))


def bench_wuzhiqi_qa_studies_family(seed: int = _SEED + 4):
    """wuzhiqi_qa_studies: synthetic correctness bench."""
    return _finite_blob(wuzhiqi_qa_studies.bench_wuzhiqi_qa_studies(seed))


def bench_xiangliu_qa_studies_family(seed: int = _SEED + 5):
    """xiangliu_qa_studies: synthetic correctness bench."""
    return _finite_blob(xiangliu_qa_studies.bench_xiangliu_qa_studies(seed))
