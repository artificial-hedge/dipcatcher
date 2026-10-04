"""Wave-1279 bench adapters: distributed-training canon (SYNTHETIC only)."""

from quant_fund.models import (
    activation_checkpoint_studies,
    fsdp_sharding_studies,
    hybrid_parallel_studies,
    pipeline_schedule_studies,
    sequence_parallel_studies,
    zero_optimizer_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12790


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_activation_checkpoint_studies_family(seed: int = _SEED + 0):
    """activation_checkpoint_studies: synthetic correctness bench."""
    return _finite_blob(activation_checkpoint_studies.bench_activation_checkpoint_studies(seed))


def bench_fsdp_sharding_studies_family(seed: int = _SEED + 1):
    """fsdp_sharding_studies: synthetic correctness bench."""
    return _finite_blob(fsdp_sharding_studies.bench_fsdp_sharding_studies(seed))


def bench_hybrid_parallel_studies_family(seed: int = _SEED + 2):
    """hybrid_parallel_studies: synthetic correctness bench."""
    return _finite_blob(hybrid_parallel_studies.bench_hybrid_parallel_studies(seed))


def bench_pipeline_schedule_studies_family(seed: int = _SEED + 3):
    """pipeline_schedule_studies: synthetic correctness bench."""
    return _finite_blob(pipeline_schedule_studies.bench_pipeline_schedule_studies(seed))


def bench_sequence_parallel_studies_family(seed: int = _SEED + 4):
    """sequence_parallel_studies: synthetic correctness bench."""
    return _finite_blob(sequence_parallel_studies.bench_sequence_parallel_studies(seed))


def bench_zero_optimizer_studies_family(seed: int = _SEED + 5):
    """zero_optimizer_studies: synthetic correctness bench."""
    return _finite_blob(zero_optimizer_studies.bench_zero_optimizer_studies(seed))
