"""Wave-1324 bench adapters: judge-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    alpacaeval_studies,
    arena_hard_studies,
    judge_bench_studies,
    mt_bench_judge_studies,
    prometheus_eval_studies,
    reward_bench_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13240


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alpacaeval_studies_family(seed: int = _SEED + 0):
    """alpacaeval_studies: synthetic correctness bench."""
    return _finite_blob(alpacaeval_studies.bench_alpacaeval_studies(seed))


def bench_arena_hard_studies_family(seed: int = _SEED + 1):
    """arena_hard_studies: synthetic correctness bench."""
    return _finite_blob(arena_hard_studies.bench_arena_hard_studies(seed))


def bench_judge_bench_studies_family(seed: int = _SEED + 2):
    """judge_bench_studies: synthetic correctness bench."""
    return _finite_blob(judge_bench_studies.bench_judge_bench_studies(seed))


def bench_mt_bench_judge_studies_family(seed: int = _SEED + 3):
    """mt_bench_judge_studies: synthetic correctness bench."""
    return _finite_blob(mt_bench_judge_studies.bench_mt_bench_judge_studies(seed))


def bench_prometheus_eval_studies_family(seed: int = _SEED + 4):
    """prometheus_eval_studies: synthetic correctness bench."""
    return _finite_blob(prometheus_eval_studies.bench_prometheus_eval_studies(seed))


def bench_reward_bench_studies_family(seed: int = _SEED + 5):
    """reward_bench_studies: synthetic correctness bench."""
    return _finite_blob(reward_bench_studies.bench_reward_bench_studies(seed))
