"""Wave-1290 bench adapters: quantization/compression canon (SYNTHETIC only)."""

from quant_fund.models import (
    awq_studies,
    entropy_code_quant_studies,
    gptq_studies,
    kv_cache_quant_studies,
    smoothquant_studies,
    weight_share_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12900


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_awq_studies_family(seed: int = _SEED + 0):
    """awq_studies: synthetic correctness bench."""
    return _finite_blob(awq_studies.bench_awq_studies(seed))


def bench_entropy_code_quant_studies_family(seed: int = _SEED + 1):
    """entropy_code_quant_studies: synthetic correctness bench."""
    return _finite_blob(entropy_code_quant_studies.bench_entropy_code_quant_studies(seed))


def bench_gptq_studies_family(seed: int = _SEED + 2):
    """gptq_studies: synthetic correctness bench."""
    return _finite_blob(gptq_studies.bench_gptq_studies(seed))


def bench_kv_cache_quant_studies_family(seed: int = _SEED + 3):
    """kv_cache_quant_studies: synthetic correctness bench."""
    return _finite_blob(kv_cache_quant_studies.bench_kv_cache_quant_studies(seed))


def bench_smoothquant_studies_family(seed: int = _SEED + 4):
    """smoothquant_studies: synthetic correctness bench."""
    return _finite_blob(smoothquant_studies.bench_smoothquant_studies(seed))


def bench_weight_share_studies_family(seed: int = _SEED + 5):
    """weight_share_studies: synthetic correctness bench."""
    return _finite_blob(weight_share_studies.bench_weight_share_studies(seed))
