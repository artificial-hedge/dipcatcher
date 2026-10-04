"""Wave-1302 bench adapters: privacy-inference canon (SYNTHETIC only)."""

from quant_fund.models import (
    canary_infer_studies,
    deep_leak_studies,
    gradient_leak_studies,
    lira_studies,
    membership_infer_studies,
    shadow_model_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13020


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_canary_infer_studies_family(seed: int = _SEED + 0):
    """canary_infer_studies: synthetic correctness bench."""
    return _finite_blob(canary_infer_studies.bench_canary_infer_studies(seed))


def bench_deep_leak_studies_family(seed: int = _SEED + 1):
    """deep_leak_studies: synthetic correctness bench."""
    return _finite_blob(deep_leak_studies.bench_deep_leak_studies(seed))


def bench_gradient_leak_studies_family(seed: int = _SEED + 2):
    """gradient_leak_studies: synthetic correctness bench."""
    return _finite_blob(gradient_leak_studies.bench_gradient_leak_studies(seed))


def bench_lira_studies_family(seed: int = _SEED + 3):
    """lira_studies: synthetic correctness bench."""
    return _finite_blob(lira_studies.bench_lira_studies(seed))


def bench_membership_infer_studies_family(seed: int = _SEED + 4):
    """membership_infer_studies: synthetic correctness bench."""
    return _finite_blob(membership_infer_studies.bench_membership_infer_studies(seed))


def bench_shadow_model_studies_family(seed: int = _SEED + 5):
    """shadow_model_studies: synthetic correctness bench."""
    return _finite_blob(shadow_model_studies.bench_shadow_model_studies(seed))
