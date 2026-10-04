"""Wave-1297 bench adapters: eval-science canon (SYNTHETIC only)."""

from quant_fund.models import (
    benchmark_gaming_studies,
    benchmark_saturate_studies,
    contamination_studies,
    eval_coverage_studies,
    eval_reliability_studies,
    lm_eval_harness_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12970


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_benchmark_gaming_studies_family(seed: int = _SEED + 0):
    """benchmark_gaming_studies: synthetic correctness bench."""
    return _finite_blob(benchmark_gaming_studies.bench_benchmark_gaming_studies(seed))


def bench_benchmark_saturate_studies_family(seed: int = _SEED + 1):
    """benchmark_saturate_studies: synthetic correctness bench."""
    return _finite_blob(benchmark_saturate_studies.bench_benchmark_saturate_studies(seed))


def bench_contamination_studies_family(seed: int = _SEED + 2):
    """contamination_studies: synthetic correctness bench."""
    return _finite_blob(contamination_studies.bench_contamination_studies(seed))


def bench_eval_coverage_studies_family(seed: int = _SEED + 3):
    """eval_coverage_studies: synthetic correctness bench."""
    return _finite_blob(eval_coverage_studies.bench_eval_coverage_studies(seed))


def bench_eval_reliability_studies_family(seed: int = _SEED + 4):
    """eval_reliability_studies: synthetic correctness bench."""
    return _finite_blob(eval_reliability_studies.bench_eval_reliability_studies(seed))


def bench_lm_eval_harness_studies_family(seed: int = _SEED + 5):
    """lm_eval_harness_studies: synthetic correctness bench."""
    return _finite_blob(lm_eval_harness_studies.bench_lm_eval_harness_studies(seed))
