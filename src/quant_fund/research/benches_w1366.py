"""Wave-1366 bench adapters: faithfulness-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    align_score_studies,
    dice_eval_studies,
    factcc_lite_studies,
    faith_eval_studies,
    quest_eval_studies,
    summa_eval_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13660


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


def bench_align_score_studies_family(seed: int = _SEED + 0):
    """align_score_studies: synthetic correctness bench."""
    return _finite_blob(align_score_studies.bench_align_score_studies(seed))


def bench_dice_eval_studies_family(seed: int = _SEED + 1):
    """dice_eval_studies: synthetic correctness bench."""
    return _finite_blob(dice_eval_studies.bench_dice_eval_studies(seed))


def bench_factcc_lite_studies_family(seed: int = _SEED + 2):
    """factcc_lite_studies: synthetic correctness bench."""
    return _finite_blob(factcc_lite_studies.bench_factcc_lite_studies(seed))


def bench_faith_eval_studies_family(seed: int = _SEED + 3):
    """faith_eval_studies: synthetic correctness bench."""
    return _finite_blob(faith_eval_studies.bench_faith_eval_studies(seed))


def bench_quest_eval_studies_family(seed: int = _SEED + 4):
    """quest_eval_studies: synthetic correctness bench."""
    return _finite_blob(quest_eval_studies.bench_quest_eval_studies(seed))


def bench_summa_eval_studies_family(seed: int = _SEED + 5):
    """summa_eval_studies: synthetic correctness bench."""
    return _finite_blob(summa_eval_studies.bench_summa_eval_studies(seed))
