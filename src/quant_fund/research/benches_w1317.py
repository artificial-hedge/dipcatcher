"""Wave-1317 bench adapters: eval-science-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    arc_eval_studies,
    do_anything_studies,
    step_eval_studies,
    strong_reject_studies,
    verifier_reward_studies,
    winogrande_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13170


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


def bench_arc_eval_studies_family(seed: int = _SEED + 0):
    """arc_eval_studies: synthetic correctness bench."""
    return _finite_blob(arc_eval_studies.bench_arc_eval_studies(seed))


def bench_do_anything_studies_family(seed: int = _SEED + 1):
    """do_anything_studies: synthetic correctness bench."""
    return _finite_blob(do_anything_studies.bench_do_anything_studies(seed))


def bench_step_eval_studies_family(seed: int = _SEED + 2):
    """step_eval_studies: synthetic correctness bench."""
    return _finite_blob(step_eval_studies.bench_step_eval_studies(seed))


def bench_strong_reject_studies_family(seed: int = _SEED + 3):
    """strong_reject_studies: synthetic correctness bench."""
    return _finite_blob(strong_reject_studies.bench_strong_reject_studies(seed))


def bench_verifier_reward_studies_family(seed: int = _SEED + 4):
    """verifier_reward_studies: synthetic correctness bench."""
    return _finite_blob(verifier_reward_studies.bench_verifier_reward_studies(seed))


def bench_winogrande_studies_family(seed: int = _SEED + 5):
    """winogrande_studies: synthetic correctness bench."""
    return _finite_blob(winogrande_studies.bench_winogrande_studies(seed))
