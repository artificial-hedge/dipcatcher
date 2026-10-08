"""Wave-1320 bench adapters: agent-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    gaia_bench_studies,
    mmbench_agent_studies,
    osworld_studies,
    screen_eval_studies,
    vsi_bench_studies,
    webvoyager_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13200


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


def bench_gaia_bench_studies_family(seed: int = _SEED + 0):
    """gaia_bench_studies: synthetic correctness bench."""
    return _finite_blob(gaia_bench_studies.bench_gaia_bench_studies(seed))


def bench_mmbench_agent_studies_family(seed: int = _SEED + 1):
    """mmbench_agent_studies: synthetic correctness bench."""
    return _finite_blob(mmbench_agent_studies.bench_mmbench_agent_studies(seed))


def bench_osworld_studies_family(seed: int = _SEED + 2):
    """osworld_studies: synthetic correctness bench."""
    return _finite_blob(osworld_studies.bench_osworld_studies(seed))


def bench_screen_eval_studies_family(seed: int = _SEED + 3):
    """screen_eval_studies: synthetic correctness bench."""
    return _finite_blob(screen_eval_studies.bench_screen_eval_studies(seed))


def bench_vsi_bench_studies_family(seed: int = _SEED + 4):
    """vsi_bench_studies: synthetic correctness bench."""
    return _finite_blob(vsi_bench_studies.bench_vsi_bench_studies(seed))


def bench_webvoyager_studies_family(seed: int = _SEED + 5):
    """webvoyager_studies: synthetic correctness bench."""
    return _finite_blob(webvoyager_studies.bench_webvoyager_studies(seed))
