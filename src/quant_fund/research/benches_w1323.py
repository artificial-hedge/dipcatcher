"""Wave-1323 bench adapters: code-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    bigcodebench_studies,
    ds1000_studies,
    humaneval_plus_studies,
    livecodebench_studies,
    mbpp_plus_studies,
    swe_perf_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13230


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


def bench_bigcodebench_studies_family(seed: int = _SEED + 0):
    """bigcodebench_studies: synthetic correctness bench."""
    return _finite_blob(bigcodebench_studies.bench_bigcodebench_studies(seed))


def bench_ds1000_studies_family(seed: int = _SEED + 1):
    """ds1000_studies: synthetic correctness bench."""
    return _finite_blob(ds1000_studies.bench_ds1000_studies(seed))


def bench_humaneval_plus_studies_family(seed: int = _SEED + 2):
    """humaneval_plus_studies: synthetic correctness bench."""
    return _finite_blob(humaneval_plus_studies.bench_humaneval_plus_studies(seed))


def bench_livecodebench_studies_family(seed: int = _SEED + 3):
    """livecodebench_studies: synthetic correctness bench."""
    return _finite_blob(livecodebench_studies.bench_livecodebench_studies(seed))


def bench_mbpp_plus_studies_family(seed: int = _SEED + 4):
    """mbpp_plus_studies: synthetic correctness bench."""
    return _finite_blob(mbpp_plus_studies.bench_mbpp_plus_studies(seed))


def bench_swe_perf_studies_family(seed: int = _SEED + 5):
    """swe_perf_studies: synthetic correctness bench."""
    return _finite_blob(swe_perf_studies.bench_swe_perf_studies(seed))
