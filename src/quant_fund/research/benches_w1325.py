"""Wave-1325 bench adapters: long-context-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    babilong_studies,
    infinitebench_studies,
    longbench_studies,
    lv_eval_studies,
    ruler_bench_studies,
    zero_scrolls_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13250


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_babilong_studies_family(seed: int = _SEED + 0):
    """babilong_studies: synthetic correctness bench."""
    return _finite_blob(babilong_studies.bench_babilong_studies(seed))


def bench_infinitebench_studies_family(seed: int = _SEED + 1):
    """infinitebench_studies: synthetic correctness bench."""
    return _finite_blob(infinitebench_studies.bench_infinitebench_studies(seed))


def bench_longbench_studies_family(seed: int = _SEED + 2):
    """longbench_studies: synthetic correctness bench."""
    return _finite_blob(longbench_studies.bench_longbench_studies(seed))


def bench_lv_eval_studies_family(seed: int = _SEED + 3):
    """lv_eval_studies: synthetic correctness bench."""
    return _finite_blob(lv_eval_studies.bench_lv_eval_studies(seed))


def bench_ruler_bench_studies_family(seed: int = _SEED + 4):
    """ruler_bench_studies: synthetic correctness bench."""
    return _finite_blob(ruler_bench_studies.bench_ruler_bench_studies(seed))


def bench_zero_scrolls_studies_family(seed: int = _SEED + 5):
    """zero_scrolls_studies: synthetic correctness bench."""
    return _finite_blob(zero_scrolls_studies.bench_zero_scrolls_studies(seed))
