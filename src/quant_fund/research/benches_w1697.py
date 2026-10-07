"""Wave-1697 bench adapters: chinese-myth-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    aoqin_qa_studies,
    guandi_qa_studies,
    houyi_qa_studies,
    wenchang_qa_studies,
    yutu_qa_studies,
    zao_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16970


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


def bench_aoqin_qa_studies_family(seed: int = _SEED + 0):
    """aoqin_qa_studies: synthetic correctness bench."""
    return _finite_blob(aoqin_qa_studies.bench_aoqin_qa_studies(seed))


def bench_guandi_qa_studies_family(seed: int = _SEED + 1):
    """guandi_qa_studies: synthetic correctness bench."""
    return _finite_blob(guandi_qa_studies.bench_guandi_qa_studies(seed))


def bench_houyi_qa_studies_family(seed: int = _SEED + 2):
    """houyi_qa_studies: synthetic correctness bench."""
    return _finite_blob(houyi_qa_studies.bench_houyi_qa_studies(seed))


def bench_wenchang_qa_studies_family(seed: int = _SEED + 3):
    """wenchang_qa_studies: synthetic correctness bench."""
    return _finite_blob(wenchang_qa_studies.bench_wenchang_qa_studies(seed))


def bench_yutu_qa_studies_family(seed: int = _SEED + 4):
    """yutu_qa_studies: synthetic correctness bench."""
    return _finite_blob(yutu_qa_studies.bench_yutu_qa_studies(seed))


def bench_zao_qa_studies_family(seed: int = _SEED + 5):
    """zao_qa_studies: synthetic correctness bench."""
    return _finite_blob(zao_qa_studies.bench_zao_qa_studies(seed))
