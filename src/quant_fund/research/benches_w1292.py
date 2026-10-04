"""Wave-1292 bench adapters: RLVR/verifiable-rewards canon (SYNTHETIC only)."""

from quant_fund.models import (
    grpo_studies,
    math_reward_studies,
    outcome_reward_studies,
    process_reward_studies,
    rlvr_studies,
    verifiable_reward_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12920


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_grpo_studies_family(seed: int = _SEED + 0):
    """grpo_studies: synthetic correctness bench."""
    return _finite_blob(grpo_studies.bench_grpo_studies(seed))


def bench_math_reward_studies_family(seed: int = _SEED + 1):
    """math_reward_studies: synthetic correctness bench."""
    return _finite_blob(math_reward_studies.bench_math_reward_studies(seed))


def bench_outcome_reward_studies_family(seed: int = _SEED + 2):
    """outcome_reward_studies: synthetic correctness bench."""
    return _finite_blob(outcome_reward_studies.bench_outcome_reward_studies(seed))


def bench_process_reward_studies_family(seed: int = _SEED + 3):
    """process_reward_studies: synthetic correctness bench."""
    return _finite_blob(process_reward_studies.bench_process_reward_studies(seed))


def bench_rlvr_studies_family(seed: int = _SEED + 4):
    """rlvr_studies: synthetic correctness bench."""
    return _finite_blob(rlvr_studies.bench_rlvr_studies(seed))


def bench_verifiable_reward_studies_family(seed: int = _SEED + 5):
    """verifiable_reward_studies: synthetic correctness bench."""
    return _finite_blob(verifiable_reward_studies.bench_verifiable_reward_studies(seed))
