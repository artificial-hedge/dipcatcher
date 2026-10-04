"""Wave-1352 bench adapters: social-bias-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    bias_bench_studies,
    crowsp_lite_studies,
    honesty_lie_studies,
    social_iqa2_studies,
    stereo_lite_studies,
    wino_bias_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13520


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bias_bench_studies_family(seed: int = _SEED + 0):
    """bias_bench_studies: synthetic correctness bench."""
    return _finite_blob(bias_bench_studies.bench_bias_bench_studies(seed))


def bench_crowsp_lite_studies_family(seed: int = _SEED + 1):
    """crowsp_lite_studies: synthetic correctness bench."""
    return _finite_blob(crowsp_lite_studies.bench_crowsp_lite_studies(seed))


def bench_honesty_lie_studies_family(seed: int = _SEED + 2):
    """honesty_lie_studies: synthetic correctness bench."""
    return _finite_blob(honesty_lie_studies.bench_honesty_lie_studies(seed))


def bench_social_iqa2_studies_family(seed: int = _SEED + 3):
    """social_iqa2_studies: synthetic correctness bench."""
    return _finite_blob(social_iqa2_studies.bench_social_iqa2_studies(seed))


def bench_stereo_lite_studies_family(seed: int = _SEED + 4):
    """stereo_lite_studies: synthetic correctness bench."""
    return _finite_blob(stereo_lite_studies.bench_stereo_lite_studies(seed))


def bench_wino_bias_studies_family(seed: int = _SEED + 5):
    """wino_bias_studies: synthetic correctness bench."""
    return _finite_blob(wino_bias_studies.bench_wino_bias_studies(seed))
