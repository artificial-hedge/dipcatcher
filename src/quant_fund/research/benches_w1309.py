"""Wave-1309 bench adapters: hard-benchmark canon (SYNTHETIC only)."""

from quant_fund.models import (
    frontier_math_studies,
    gpqa_studies,
    hle_studies,
    mmlu_pro_studies,
    tau_bench_studies,
    workarena_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13090


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_frontier_math_studies_family(seed: int = _SEED + 0):
    """frontier_math_studies: synthetic correctness bench."""
    return _finite_blob(frontier_math_studies.bench_frontier_math_studies(seed))


def bench_gpqa_studies_family(seed: int = _SEED + 1):
    """gpqa_studies: synthetic correctness bench."""
    return _finite_blob(gpqa_studies.bench_gpqa_studies(seed))


def bench_hle_studies_family(seed: int = _SEED + 2):
    """hle_studies: synthetic correctness bench."""
    return _finite_blob(hle_studies.bench_hle_studies(seed))


def bench_mmlu_pro_studies_family(seed: int = _SEED + 3):
    """mmlu_pro_studies: synthetic correctness bench."""
    return _finite_blob(mmlu_pro_studies.bench_mmlu_pro_studies(seed))


def bench_tau_bench_studies_family(seed: int = _SEED + 4):
    """tau_bench_studies: synthetic correctness bench."""
    return _finite_blob(tau_bench_studies.bench_tau_bench_studies(seed))


def bench_workarena_studies_family(seed: int = _SEED + 5):
    """workarena_studies: synthetic correctness bench."""
    return _finite_blob(workarena_studies.bench_workarena_studies(seed))
