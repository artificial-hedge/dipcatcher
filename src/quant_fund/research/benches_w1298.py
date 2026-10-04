"""Wave-1298 bench adapters: agentic-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    browse_eval_studies,
    os_world_studies,
    swe_bench_studies,
    terminal_bench_studies,
    tool_use_eval_studies,
    web_arena_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12980


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_browse_eval_studies_family(seed: int = _SEED + 0):
    """browse_eval_studies: synthetic correctness bench."""
    return _finite_blob(browse_eval_studies.bench_browse_eval_studies(seed))


def bench_os_world_studies_family(seed: int = _SEED + 1):
    """os_world_studies: synthetic correctness bench."""
    return _finite_blob(os_world_studies.bench_os_world_studies(seed))


def bench_swe_bench_studies_family(seed: int = _SEED + 2):
    """swe_bench_studies: synthetic correctness bench."""
    return _finite_blob(swe_bench_studies.bench_swe_bench_studies(seed))


def bench_terminal_bench_studies_family(seed: int = _SEED + 3):
    """terminal_bench_studies: synthetic correctness bench."""
    return _finite_blob(terminal_bench_studies.bench_terminal_bench_studies(seed))


def bench_tool_use_eval_studies_family(seed: int = _SEED + 4):
    """tool_use_eval_studies: synthetic correctness bench."""
    return _finite_blob(tool_use_eval_studies.bench_tool_use_eval_studies(seed))


def bench_web_arena_studies_family(seed: int = _SEED + 5):
    """web_arena_studies: synthetic correctness bench."""
    return _finite_blob(web_arena_studies.bench_web_arena_studies(seed))
