"""Wave-1384 bench adapters: live-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    aider_polyglot_studies,
    hum_eval_studies,
    livebench_arena_studies,
    mbti_eval_studies,
    olmes_lite_studies,
    plus_eval_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13840


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aider_polyglot_studies_family(seed: int = _SEED + 0):
    """aider_polyglot_studies: synthetic correctness bench."""
    return _finite_blob(aider_polyglot_studies.bench_aider_polyglot_studies(seed))


def bench_hum_eval_studies_family(seed: int = _SEED + 1):
    """hum_eval_studies: synthetic correctness bench."""
    return _finite_blob(hum_eval_studies.bench_hum_eval_studies(seed))


def bench_livebench_arena_studies_family(seed: int = _SEED + 2):
    """livebench_arena_studies: synthetic correctness bench."""
    return _finite_blob(livebench_arena_studies.bench_livebench_arena_studies(seed))


def bench_mbti_eval_studies_family(seed: int = _SEED + 3):
    """mbti_eval_studies: synthetic correctness bench."""
    return _finite_blob(mbti_eval_studies.bench_mbti_eval_studies(seed))


def bench_olmes_lite_studies_family(seed: int = _SEED + 4):
    """olmes_lite_studies: synthetic correctness bench."""
    return _finite_blob(olmes_lite_studies.bench_olmes_lite_studies(seed))


def bench_plus_eval_studies_family(seed: int = _SEED + 5):
    """plus_eval_studies: synthetic correctness bench."""
    return _finite_blob(plus_eval_studies.bench_plus_eval_studies(seed))
