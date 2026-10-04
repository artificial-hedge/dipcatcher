"""Wave-1266 bench adapters: LLM-inference-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    diffusion_lm_studies,
    kv_compression_studies,
    medusa_speculation_studies,
    moe_shared_expert_studies,
    rope_scaling_studies,
    sparse_attention_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12660


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_diffusion_lm_studies_family(seed: int = _SEED + 0):
    """diffusion_lm_studies: synthetic correctness bench."""
    return _finite_blob(diffusion_lm_studies.bench_diffusion_lm_studies(seed))


def bench_kv_compression_studies_family(seed: int = _SEED + 1):
    """kv_compression_studies: synthetic correctness bench."""
    return _finite_blob(kv_compression_studies.bench_kv_compression_studies(seed))


def bench_medusa_speculation_studies_family(seed: int = _SEED + 2):
    """medusa_speculation_studies: synthetic correctness bench."""
    return _finite_blob(medusa_speculation_studies.bench_medusa_speculation_studies(seed))


def bench_moe_shared_expert_studies_family(seed: int = _SEED + 3):
    """moe_shared_expert_studies: synthetic correctness bench."""
    return _finite_blob(moe_shared_expert_studies.bench_moe_shared_expert_studies(seed))


def bench_rope_scaling_studies_family(seed: int = _SEED + 4):
    """rope_scaling_studies: synthetic correctness bench."""
    return _finite_blob(rope_scaling_studies.bench_rope_scaling_studies(seed))


def bench_sparse_attention_studies_family(seed: int = _SEED + 5):
    """sparse_attention_studies: synthetic correctness bench."""
    return _finite_blob(sparse_attention_studies.bench_sparse_attention_studies(seed))
