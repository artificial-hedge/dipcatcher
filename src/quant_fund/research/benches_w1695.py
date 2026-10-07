"""Wave-1695 bench adapters: chinese-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    dijiang_qa_studies,
    huli_qa_studies,
    jiangshi_qa_studies,
    mogwai_qa_studies,
    yaoguai_qa_studies,
    zhuyin_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16950


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_dijiang_qa_studies_family(seed: int = _SEED + 0):
    """dijiang_qa_studies: synthetic correctness bench."""
    return _finite_blob(dijiang_qa_studies.bench_dijiang_qa_studies(seed))


def bench_huli_qa_studies_family(seed: int = _SEED + 1):
    """huli_qa_studies: synthetic correctness bench."""
    return _finite_blob(huli_qa_studies.bench_huli_qa_studies(seed))


def bench_jiangshi_qa_studies_family(seed: int = _SEED + 2):
    """jiangshi_qa_studies: synthetic correctness bench."""
    return _finite_blob(jiangshi_qa_studies.bench_jiangshi_qa_studies(seed))


def bench_mogwai_qa_studies_family(seed: int = _SEED + 3):
    """mogwai_qa_studies: synthetic correctness bench."""
    return _finite_blob(mogwai_qa_studies.bench_mogwai_qa_studies(seed))


def bench_yaoguai_qa_studies_family(seed: int = _SEED + 4):
    """yaoguai_qa_studies: synthetic correctness bench."""
    return _finite_blob(yaoguai_qa_studies.bench_yaoguai_qa_studies(seed))


def bench_zhuyin_qa_studies_family(seed: int = _SEED + 5):
    """zhuyin_qa_studies: synthetic correctness bench."""
    return _finite_blob(zhuyin_qa_studies.bench_zhuyin_qa_studies(seed))
