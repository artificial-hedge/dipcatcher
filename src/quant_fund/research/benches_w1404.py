"""Wave-1404 bench adapters: video-QA canon (SYNTHETIC only)."""

from quant_fund.models import (
    activitynet_qa_studies,
    how2qa_lite_studies,
    movie_qa_lite_studies,
    msrvtt_qa_studies,
    nextqa_lite_studies,
    star_qa_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14040


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


def bench_activitynet_qa_studies_family(seed: int = _SEED + 0):
    """activitynet_qa_studies: synthetic correctness bench."""
    return _finite_blob(activitynet_qa_studies.bench_activitynet_qa_studies(seed))


def bench_how2qa_lite_studies_family(seed: int = _SEED + 1):
    """how2qa_lite_studies: synthetic correctness bench."""
    return _finite_blob(how2qa_lite_studies.bench_how2qa_lite_studies(seed))


def bench_movie_qa_lite_studies_family(seed: int = _SEED + 2):
    """movie_qa_lite_studies: synthetic correctness bench."""
    return _finite_blob(movie_qa_lite_studies.bench_movie_qa_lite_studies(seed))


def bench_msrvtt_qa_studies_family(seed: int = _SEED + 3):
    """msrvtt_qa_studies: synthetic correctness bench."""
    return _finite_blob(msrvtt_qa_studies.bench_msrvtt_qa_studies(seed))


def bench_nextqa_lite_studies_family(seed: int = _SEED + 4):
    """nextqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(nextqa_lite_studies.bench_nextqa_lite_studies(seed))


def bench_star_qa_lite_studies_family(seed: int = _SEED + 5):
    """star_qa_lite_studies: synthetic correctness bench."""
    return _finite_blob(star_qa_lite_studies.bench_star_qa_lite_studies(seed))
