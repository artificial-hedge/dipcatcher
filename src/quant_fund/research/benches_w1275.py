"""Wave-1275 bench adapters: inference-scaling canon (SYNTHETIC only)."""

from quant_fund.models import (
    deliberate_search_studies,
    latent_reasoning_studies,
    self_improvement_studies,
    test_time_scaling_studies,
    tree_thought_studies,
    verifier_gated_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12750


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_deliberate_search_studies_family(seed: int = _SEED + 0):
    """deliberate_search_studies: synthetic correctness bench."""
    return _finite_blob(deliberate_search_studies.bench_deliberate_search_studies(seed))


def bench_latent_reasoning_studies_family(seed: int = _SEED + 1):
    """latent_reasoning_studies: synthetic correctness bench."""
    return _finite_blob(latent_reasoning_studies.bench_latent_reasoning_studies(seed))


def bench_self_improvement_studies_family(seed: int = _SEED + 2):
    """self_improvement_studies: synthetic correctness bench."""
    return _finite_blob(self_improvement_studies.bench_self_improvement_studies(seed))


def bench_test_time_scaling_studies_family(seed: int = _SEED + 3):
    """test_time_scaling_studies: synthetic correctness bench."""
    return _finite_blob(test_time_scaling_studies.bench_test_time_scaling_studies(seed))


def bench_tree_thought_studies_family(seed: int = _SEED + 4):
    """tree_thought_studies: synthetic correctness bench."""
    return _finite_blob(tree_thought_studies.bench_tree_thought_studies(seed))


def bench_verifier_gated_studies_family(seed: int = _SEED + 5):
    """verifier_gated_studies: synthetic correctness bench."""
    return _finite_blob(verifier_gated_studies.bench_verifier_gated_studies(seed))
