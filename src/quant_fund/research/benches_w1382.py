"""Wave-1382 bench adapters: LLM-eval-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    fact_score_studies,
    gpt_score_studies,
    helm_lite_studies,
    lmsys_eval_studies,
    nugget_eval_studies,
    vicuna_bench_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13820


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


def bench_fact_score_studies_family(seed: int = _SEED + 0):
    """fact_score_studies: synthetic correctness bench."""
    return _finite_blob(fact_score_studies.bench_fact_score_studies(seed))


def bench_gpt_score_studies_family(seed: int = _SEED + 1):
    """gpt_score_studies: synthetic correctness bench."""
    return _finite_blob(gpt_score_studies.bench_gpt_score_studies(seed))


def bench_helm_lite_studies_family(seed: int = _SEED + 2):
    """helm_lite_studies: synthetic correctness bench."""
    return _finite_blob(helm_lite_studies.bench_helm_lite_studies(seed))


def bench_lmsys_eval_studies_family(seed: int = _SEED + 3):
    """lmsys_eval_studies: synthetic correctness bench."""
    return _finite_blob(lmsys_eval_studies.bench_lmsys_eval_studies(seed))


def bench_nugget_eval_studies_family(seed: int = _SEED + 4):
    """nugget_eval_studies: synthetic correctness bench."""
    return _finite_blob(nugget_eval_studies.bench_nugget_eval_studies(seed))


def bench_vicuna_bench_studies_family(seed: int = _SEED + 5):
    """vicuna_bench_studies: synthetic correctness bench."""
    return _finite_blob(vicuna_bench_studies.bench_vicuna_bench_studies(seed))
