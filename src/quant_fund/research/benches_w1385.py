"""Wave-1385 bench adapters: web-agent canon (SYNTHETIC only)."""

from quant_fund.models import (
    airtasks_studies,
    browsergym_studies,
    maze_eval_studies,
    mmind2web_studies,
    screenqa_studies,
    weblinx_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13850


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


def bench_airtasks_studies_family(seed: int = _SEED + 0):
    """airtasks_studies: synthetic correctness bench."""
    return _finite_blob(airtasks_studies.bench_airtasks_studies(seed))


def bench_browsergym_studies_family(seed: int = _SEED + 1):
    """browsergym_studies: synthetic correctness bench."""
    return _finite_blob(browsergym_studies.bench_browsergym_studies(seed))


def bench_maze_eval_studies_family(seed: int = _SEED + 2):
    """maze_eval_studies: synthetic correctness bench."""
    return _finite_blob(maze_eval_studies.bench_maze_eval_studies(seed))


def bench_mmind2web_studies_family(seed: int = _SEED + 3):
    """mmind2web_studies: synthetic correctness bench."""
    return _finite_blob(mmind2web_studies.bench_mmind2web_studies(seed))


def bench_screenqa_studies_family(seed: int = _SEED + 4):
    """screenqa_studies: synthetic correctness bench."""
    return _finite_blob(screenqa_studies.bench_screenqa_studies(seed))


def bench_weblinx_studies_family(seed: int = _SEED + 5):
    """weblinx_studies: synthetic correctness bench."""
    return _finite_blob(weblinx_studies.bench_weblinx_studies(seed))
