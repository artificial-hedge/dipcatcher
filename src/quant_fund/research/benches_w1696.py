"""Wave-1696 bench adapters: chinese-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    dongwanggong_qa_studies,
    fuxi_qa_studies,
    kuafu_qa_studies,
    nuwa_qa_studies,
    shennong_qa_studies,
    xiwangmu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16960


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


def bench_dongwanggong_qa_studies_family(seed: int = _SEED + 0):
    """dongwanggong_qa_studies: synthetic correctness bench."""
    return _finite_blob(dongwanggong_qa_studies.bench_dongwanggong_qa_studies(seed))


def bench_fuxi_qa_studies_family(seed: int = _SEED + 1):
    """fuxi_qa_studies: synthetic correctness bench."""
    return _finite_blob(fuxi_qa_studies.bench_fuxi_qa_studies(seed))


def bench_kuafu_qa_studies_family(seed: int = _SEED + 2):
    """kuafu_qa_studies: synthetic correctness bench."""
    return _finite_blob(kuafu_qa_studies.bench_kuafu_qa_studies(seed))


def bench_nuwa_qa_studies_family(seed: int = _SEED + 3):
    """nuwa_qa_studies: synthetic correctness bench."""
    return _finite_blob(nuwa_qa_studies.bench_nuwa_qa_studies(seed))


def bench_shennong_qa_studies_family(seed: int = _SEED + 4):
    """shennong_qa_studies: synthetic correctness bench."""
    return _finite_blob(shennong_qa_studies.bench_shennong_qa_studies(seed))


def bench_xiwangmu_qa_studies_family(seed: int = _SEED + 5):
    """xiwangmu_qa_studies: synthetic correctness bench."""
    return _finite_blob(xiwangmu_qa_studies.bench_xiwangmu_qa_studies(seed))
