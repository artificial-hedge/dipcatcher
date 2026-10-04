"""Wave-1278 bench adapters: LLM-serving canon (SYNTHETIC only)."""

from quant_fund.models import (
    chunked_prefill_studies,
    continuous_batching_studies,
    disaggregated_serving_studies,
    early_exit_studies,
    prefix_caching_studies,
    tensor_parallel_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12780


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_chunked_prefill_studies_family(seed: int = _SEED + 0):
    """chunked_prefill_studies: synthetic correctness bench."""
    return _finite_blob(chunked_prefill_studies.bench_chunked_prefill_studies(seed))


def bench_continuous_batching_studies_family(seed: int = _SEED + 1):
    """continuous_batching_studies: synthetic correctness bench."""
    return _finite_blob(continuous_batching_studies.bench_continuous_batching_studies(seed))


def bench_disaggregated_serving_studies_family(seed: int = _SEED + 2):
    """disaggregated_serving_studies: synthetic correctness bench."""
    return _finite_blob(disaggregated_serving_studies.bench_disaggregated_serving_studies(seed))


def bench_early_exit_studies_family(seed: int = _SEED + 3):
    """early_exit_studies: synthetic correctness bench."""
    return _finite_blob(early_exit_studies.bench_early_exit_studies(seed))


def bench_prefix_caching_studies_family(seed: int = _SEED + 4):
    """prefix_caching_studies: synthetic correctness bench."""
    return _finite_blob(prefix_caching_studies.bench_prefix_caching_studies(seed))


def bench_tensor_parallel_studies_family(seed: int = _SEED + 5):
    """tensor_parallel_studies: synthetic correctness bench."""
    return _finite_blob(tensor_parallel_studies.bench_tensor_parallel_studies(seed))
