"""Wave-1394 bench adapters: code-agent canon (SYNTHETIC only)."""

from quant_fund.models import (
    api_eval_studies,
    apps_lite_studies,
    livecode_studies,
    mbpp_lite_studies,
    restbench_studies,
    swe_gym_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13940


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_api_eval_studies_family(seed: int = _SEED + 0):
    """api_eval_studies: synthetic correctness bench."""
    return _finite_blob(api_eval_studies.bench_api_eval_studies(seed))


def bench_apps_lite_studies_family(seed: int = _SEED + 1):
    """apps_lite_studies: synthetic correctness bench."""
    return _finite_blob(apps_lite_studies.bench_apps_lite_studies(seed))


def bench_livecode_studies_family(seed: int = _SEED + 2):
    """livecode_studies: synthetic correctness bench."""
    return _finite_blob(livecode_studies.bench_livecode_studies(seed))


def bench_mbpp_lite_studies_family(seed: int = _SEED + 3):
    """mbpp_lite_studies: synthetic correctness bench."""
    return _finite_blob(mbpp_lite_studies.bench_mbpp_lite_studies(seed))


def bench_restbench_studies_family(seed: int = _SEED + 4):
    """restbench_studies: synthetic correctness bench."""
    return _finite_blob(restbench_studies.bench_restbench_studies(seed))


def bench_swe_gym_studies_family(seed: int = _SEED + 5):
    """swe_gym_studies: synthetic correctness bench."""
    return _finite_blob(swe_gym_studies.bench_swe_gym_studies(seed))
