"""Wave-1336 bench adapters: agentic-eval-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    assistantbench_studies,
    mind2web_studies,
    miniwob_studies,
    visual_web_studies,
    web_nav_studies,
    webarena_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13360


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


def bench_assistantbench_studies_family(seed: int = _SEED + 0):
    """assistantbench_studies: synthetic correctness bench."""
    return _finite_blob(assistantbench_studies.bench_assistantbench_studies(seed))


def bench_mind2web_studies_family(seed: int = _SEED + 1):
    """mind2web_studies: synthetic correctness bench."""
    return _finite_blob(mind2web_studies.bench_mind2web_studies(seed))


def bench_miniwob_studies_family(seed: int = _SEED + 2):
    """miniwob_studies: synthetic correctness bench."""
    return _finite_blob(miniwob_studies.bench_miniwob_studies(seed))


def bench_visual_web_studies_family(seed: int = _SEED + 3):
    """visual_web_studies: synthetic correctness bench."""
    return _finite_blob(visual_web_studies.bench_visual_web_studies(seed))


def bench_web_nav_studies_family(seed: int = _SEED + 4):
    """web_nav_studies: synthetic correctness bench."""
    return _finite_blob(web_nav_studies.bench_web_nav_studies(seed))


def bench_webarena_studies_family(seed: int = _SEED + 5):
    """webarena_studies: synthetic correctness bench."""
    return _finite_blob(webarena_studies.bench_webarena_studies(seed))
