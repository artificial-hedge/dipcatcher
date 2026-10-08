"""Wave-1296 bench adapters: reward-modeling-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ensemble_rm_studies,
    judge_reward_studies,
    margin_reward_studies,
    reward_hacking_studies,
    reward_uncertainty_studies,
    rm_btd_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12960


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


def bench_ensemble_rm_studies_family(seed: int = _SEED + 0):
    """ensemble_rm_studies: synthetic correctness bench."""
    return _finite_blob(ensemble_rm_studies.bench_ensemble_rm_studies(seed))


def bench_judge_reward_studies_family(seed: int = _SEED + 1):
    """judge_reward_studies: synthetic correctness bench."""
    return _finite_blob(judge_reward_studies.bench_judge_reward_studies(seed))


def bench_margin_reward_studies_family(seed: int = _SEED + 2):
    """margin_reward_studies: synthetic correctness bench."""
    return _finite_blob(margin_reward_studies.bench_margin_reward_studies(seed))


def bench_reward_hacking_studies_family(seed: int = _SEED + 3):
    """reward_hacking_studies: synthetic correctness bench."""
    return _finite_blob(reward_hacking_studies.bench_reward_hacking_studies(seed))


def bench_reward_uncertainty_studies_family(seed: int = _SEED + 4):
    """reward_uncertainty_studies: synthetic correctness bench."""
    return _finite_blob(reward_uncertainty_studies.bench_reward_uncertainty_studies(seed))


def bench_rm_btd_studies_family(seed: int = _SEED + 5):
    """rm_btd_studies: synthetic correctness bench."""
    return _finite_blob(rm_btd_studies.bench_rm_btd_studies(seed))
