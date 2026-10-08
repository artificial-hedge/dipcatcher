"""Wave-1392 bench adapters: toolbench canon (SYNTHETIC only)."""

from quant_fund.models import (
    meta_tool_studies,
    nest_tools_studies,
    toolbench2_studies,
    toolqa_lite_studies,
    ultra_tool_studies,
    work_plus_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13920


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


def bench_meta_tool_studies_family(seed: int = _SEED + 0):
    """meta_tool_studies: synthetic correctness bench."""
    return _finite_blob(meta_tool_studies.bench_meta_tool_studies(seed))


def bench_nest_tools_studies_family(seed: int = _SEED + 1):
    """nest_tools_studies: synthetic correctness bench."""
    return _finite_blob(nest_tools_studies.bench_nest_tools_studies(seed))


def bench_toolbench2_studies_family(seed: int = _SEED + 2):
    """toolbench2_studies: synthetic correctness bench."""
    return _finite_blob(toolbench2_studies.bench_toolbench2_studies(seed))


def bench_toolqa_lite_studies_family(seed: int = _SEED + 3):
    """toolqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(toolqa_lite_studies.bench_toolqa_lite_studies(seed))


def bench_ultra_tool_studies_family(seed: int = _SEED + 4):
    """ultra_tool_studies: synthetic correctness bench."""
    return _finite_blob(ultra_tool_studies.bench_ultra_tool_studies(seed))


def bench_work_plus_studies_family(seed: int = _SEED + 5):
    """work_plus_studies: synthetic correctness bench."""
    return _finite_blob(work_plus_studies.bench_work_plus_studies(seed))
