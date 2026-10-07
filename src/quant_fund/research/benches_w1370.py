"""Wave-1370 bench adapters: challenge-benchmark canon (SYNTHETIC only)."""

from quant_fund.models import (
    bbh_lite_studies,
    gpqa_lite_studies,
    if_eval_studies,
    live_bench_studies,
    olympic_bench_studies,
    trivia_qa_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13700


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


def bench_bbh_lite_studies_family(seed: int = _SEED + 0):
    """bbh_lite_studies: synthetic correctness bench."""
    return _finite_blob(bbh_lite_studies.bench_bbh_lite_studies(seed))


def bench_gpqa_lite_studies_family(seed: int = _SEED + 1):
    """gpqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(gpqa_lite_studies.bench_gpqa_lite_studies(seed))


def bench_if_eval_studies_family(seed: int = _SEED + 2):
    """if_eval_studies: synthetic correctness bench."""
    return _finite_blob(if_eval_studies.bench_if_eval_studies(seed))


def bench_live_bench_studies_family(seed: int = _SEED + 3):
    """live_bench_studies: synthetic correctness bench."""
    return _finite_blob(live_bench_studies.bench_live_bench_studies(seed))


def bench_olympic_bench_studies_family(seed: int = _SEED + 4):
    """olympic_bench_studies: synthetic correctness bench."""
    return _finite_blob(olympic_bench_studies.bench_olympic_bench_studies(seed))


def bench_trivia_qa_lite_studies_family(seed: int = _SEED + 5):
    """trivia_qa_lite_studies: synthetic correctness bench."""
    return _finite_blob(trivia_qa_lite_studies.bench_trivia_qa_lite_studies(seed))
